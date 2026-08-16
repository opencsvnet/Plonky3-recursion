//! Value-free batch-STARK shape descriptor and profile-normalization helpers.
//!
//! A [`BatchStarkShape`] records every structure currently read from a concrete
//! [`BatchStarkProof`] while constructing a recursive verifier — raw row counts,
//! table packing, ALU/extension parameters, the ordered NPO manifest, preprocessed
//! instance metadata, opened-value layout, and (optionally) FRI query layout —
//! and **none** of the proof or verification-key values (commitments, openings,
//! const-table literals, public values).
//!
//! Exact equality against a real proof is the packing / native-verify gate:
//! a proof that does not match the descriptor is rejected before values are
//! packed or verified. [`pad_to_profile`] never silently selects a larger
//! profile; overflow is a hard error. [`pad_proof_inputs_to_profile`] lifts
//! a concrete proof's traces and committed preprocessed multiplicities onto
//! that frozen profile with AIR-valid dummy rows.

use alloc::format;
use alloc::vec::Vec;

use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::{NonPrimitivePreprocessedMap, NpoTypeId};
use p3_circuit::tables::{NpoPadError, Traces};
use p3_circuit::types::WitnessId;
use p3_commit::{Mmcs, Pcs};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::FriProof;
use serde::{Deserialize, Serialize};

use crate::batch_stark_prover::{
    AirVariant, BatchStarkProof, CircuitProverData, PrimitiveTable, ProofMetadataError, RowCounts,
    TablePacking,
};

/// Value-free description of one non-primitive table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpoShapeEntry {
    /// Operation type, in proof order.
    pub op_type: NpoTypeId,
    /// Raw (pre-pad) logical row count.
    pub rows: usize,
    /// Packed lanes per AIR row.
    pub lanes: usize,
    /// Number of public values this table exposes (not the values themselves).
    pub public_values_len: usize,
    /// AIR variant tag for this table.
    pub air_variant: AirVariant,
}

/// Value-free preprocessed-instance metadata (no commitment).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreprocessedInstanceShape {
    /// Index of the committed preprocessed matrix for this instance.
    pub matrix_index: usize,
    /// Preprocessed width.
    pub width: usize,
    /// `log2` of the preprocessed domain size.
    pub degree_bits: usize,
}

/// Value-free global preprocessed layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreprocessedShape {
    /// Per-instance metadata; `None` when that instance has no preprocessed columns.
    pub instances: Vec<Option<PreprocessedInstanceShape>>,
    /// Mapping from preprocessed matrix index to instance index.
    pub matrix_to_instance: Vec<usize>,
}

/// Value-free opened-value / degree / lookup-terminal layout for one instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceOpenedShape {
    /// `log2` of the extended trace domain size.
    pub degree_bits: usize,
    /// Main-trace local opening width.
    pub trace_local: usize,
    /// Main-trace next-row opening width, if the AIR opens the next row.
    pub trace_next: Option<usize>,
    /// Preprocessed local opening width.
    pub preprocessed_local: Option<usize>,
    /// Preprocessed next-row opening width.
    pub preprocessed_next: Option<usize>,
    /// Per-chunk quotient opening widths.
    pub quotient_chunks: Vec<usize>,
    /// Random-polynomial opening width, when ZK is enabled.
    pub random: Option<usize>,
    /// Permutation-polynomial local opening width.
    pub perm_local: usize,
    /// Permutation-polynomial next-row opening width.
    pub perm_next: usize,
    /// Whether this instance carries a lookup terminal.
    pub has_lookup_terminal: bool,
}

/// Value-free FRI commit-phase / query layout.
///
/// Extracted from a [`FriProof`] via [`FriQueryShape::from_fri_proof`]. The
/// sibling *values* and Merkle paths are omitted; only counts and arities stay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FriQueryShape {
    /// Number of FRI commit-phase commitments.
    pub commit_phase_len: usize,
    /// Number of per-commit-phase grinding witnesses.
    pub commit_pow_count: usize,
    /// Number of FRI queries.
    pub query_count: usize,
    /// Length of the final polynomial.
    pub final_poly_len: usize,
    /// Per commit-phase step `log2(arity)`, taken from the first query.
    pub log_arities: Vec<usize>,
    /// Per query, per commit-phase step: number of sibling values.
    pub sibling_counts: Vec<Vec<usize>>,
}

impl FriQueryShape {
    /// Extract the query layout from a concrete FRI proof. No field values are copied.
    pub fn from_fri_proof<F, M, W, I>(proof: &FriProof<F, M, W, I>) -> Self
    where
        F: Field,
        M: Mmcs<F>,
    {
        let log_arities = proof
            .query_proofs
            .first()
            .map(|query| {
                query
                    .commit_phase_openings
                    .iter()
                    .map(|step| step.log_arity as usize)
                    .collect()
            })
            .unwrap_or_default();
        let sibling_counts = proof
            .query_proofs
            .iter()
            .map(|query| {
                query
                    .commit_phase_openings
                    .iter()
                    .map(|step| step.sibling_values.len())
                    .collect()
            })
            .collect();
        Self {
            commit_phase_len: proof.commit_phase_commits.len(),
            commit_pow_count: proof.commit_pow_witnesses.len(),
            query_count: proof.query_proofs.len(),
            final_poly_len: proof.final_poly.len(),
            log_arities,
            sibling_counts,
        }
    }
}

/// Opening proofs that expose a value-free FRI query layout.
///
/// Implemented for [`FriProof`]. Callers with a `TwoAdicFriPcs` config can
/// extract the layout without naming the concrete Merkle/witness types.
pub trait FriQueryShapeSource {
    /// Value-free FRI commit-phase / query layout of this opening proof.
    fn fri_query_shape(&self) -> FriQueryShape;
}

impl<F, M, W, I> FriQueryShapeSource for FriProof<F, M, W, I>
where
    F: Field,
    M: Mmcs<F>,
{
    fn fri_query_shape(&self) -> FriQueryShape {
        FriQueryShape::from_fri_proof(self)
    }
}

/// Value-free, canonical descriptor of a batch-STARK proof's verifier-relevant structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de> + Copy"))]
pub struct BatchStarkShape<F: Copy> {
    /// Raw primitive row counts (`Const`, `Public`, `Alu`).
    pub rows: RowCounts,
    /// Table packing (lanes, min height, Horner pack, NPO lane overrides).
    pub table_packing: TablePacking,
    /// Primitive ALU variant.
    pub alu_variant: AirVariant,
    /// Trace extension degree.
    pub ext_degree: usize,
    /// Binomial `W` when `ext_degree > 1`. This is a field *parameter*, not a
    /// proof-carried witness value.
    pub w_binomial: Option<F>,
    /// Quintic-trinomial ALU reduction flag.
    pub alu_quintic_trinomial: bool,
    /// Ordered non-primitive manifest (no public-value contents).
    pub non_primitives: Vec<NpoShapeEntry>,
    /// Preprocessed instance metadata, without the commitment.
    pub preprocessed: Option<PreprocessedShape>,
    /// Per-instance opened-value / degree / lookup-terminal layout.
    pub instances: Vec<InstanceOpenedShape>,
    /// Whether the batch carries a permutation (lookup) commitment.
    pub has_permutation_commitment: bool,
    /// Whether the batch carries a ZK randomization commitment.
    pub has_random_commitment: bool,
    /// Optional FRI query layout. `None` when the caller has not attached it.
    pub fri: Option<FriQueryShape>,
}

impl<F: Copy + PartialEq> BatchStarkShape<F> {
    /// Extract a value-free shape from a concrete proof (metadata + opened layout).
    ///
    /// FRI query layout is left `None`; attach it with [`Self::with_fri`] after
    /// extracting a [`FriQueryShape`] from the PCS opening proof.
    pub fn from_proof<SC>(proof: &BatchStarkProof<SC>) -> Self
    where
        SC: StarkGenericConfig,
        F: From<Val<SC>>,
        Val<SC>: Copy,
    {
        let non_primitives = proof
            .non_primitives
            .iter()
            .map(|entry| NpoShapeEntry {
                op_type: entry.op_type.clone(),
                rows: entry.rows,
                lanes: entry.lanes,
                public_values_len: entry.public_values.len(),
                air_variant: entry.air_variant,
            })
            .collect();

        let preprocessed = proof
            .stark_common
            .preprocessed
            .as_ref()
            .map(|gp| PreprocessedShape {
                instances: gp
                    .instances
                    .iter()
                    .map(|opt| {
                        opt.as_ref().map(|meta| PreprocessedInstanceShape {
                            matrix_index: meta.matrix_index,
                            width: meta.width,
                            degree_bits: meta.degree_bits,
                        })
                    })
                    .collect(),
                matrix_to_instance: gp.matrix_to_instance.clone(),
            });

        let instances = proof
            .proof
            .opened_values
            .instances
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let base = &inst.base_opened_values;
                InstanceOpenedShape {
                    degree_bits: proof.proof.degree_bits.get(i).copied().unwrap_or(0),
                    trace_local: base.trace_local.len(),
                    trace_next: base.trace_next.as_ref().map(Vec::len),
                    preprocessed_local: base.preprocessed_local.as_ref().map(Vec::len),
                    preprocessed_next: base.preprocessed_next.as_ref().map(Vec::len),
                    quotient_chunks: base.quotient_chunks.iter().map(Vec::len).collect(),
                    random: base.random.as_ref().map(Vec::len),
                    perm_local: inst.permutation_local.len(),
                    perm_next: inst.permutation_next.len(),
                    has_lookup_terminal: proof
                        .proof
                        .lookup_terminals
                        .get(i)
                        .is_some_and(Option::is_some),
                }
            })
            .collect();

        Self {
            rows: proof.rows,
            table_packing: proof.table_packing.clone(),
            alu_variant: proof.alu_variant,
            ext_degree: proof.ext_degree,
            w_binomial: proof.w_binomial.map(F::from),
            alu_quintic_trinomial: proof.alu_quintic_trinomial,
            non_primitives,
            preprocessed,
            instances,
            has_permutation_commitment: proof.proof.commitments.permutation.is_some(),
            has_random_commitment: proof.proof.commitments.random.is_some(),
            fri: None,
        }
    }

    /// Attach a FRI query layout extracted from the PCS opening proof.
    #[must_use]
    pub fn with_fri(mut self, fri: FriQueryShape) -> Self {
        self.fri = Some(fri);
        self
    }

    /// Extract metadata **and** the FRI query layout from a `TwoAdicFriPcs` proof.
    pub fn from_proof_with_fri<SC>(proof: &BatchStarkProof<SC>) -> Self
    where
        SC: StarkGenericConfig,
        F: From<Val<SC>>,
        Val<SC>: Copy,
        <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof: FriQueryShapeSource,
    {
        Self::from_proof(proof).with_fri(proof.proof.opening_proof.fri_query_shape())
    }

    /// Exact structural equality of this descriptor against a concrete proof.
    ///
    /// Metadata and opened-value layout are always compared. FRI is compared
    /// only when `self.fri` is `None` (skipped) — [`from_proof`] cannot see the
    /// generic PCS opening proof. Use [`Self::matches_proof_with_fri`] when the
    /// descriptor locks a FRI layout.
    pub fn matches_proof<SC>(&self, proof: &BatchStarkProof<SC>) -> Result<(), ProofMetadataError>
    where
        SC: StarkGenericConfig,
        F: From<Val<SC>>,
        Val<SC>: Copy + PartialEq,
    {
        let got = Self::from_proof(proof);
        self.matches_shape(&got)
    }

    /// Exact structural equality including FRI query layout.
    ///
    /// Requires a `TwoAdicFriPcs` opening proof. A descriptor that locks `fri`
    /// is compared against the proof's extracted layout; a mutation of query
    /// count, commit-phase length, or sibling arity rejects.
    pub fn matches_proof_with_fri<SC>(
        &self,
        proof: &BatchStarkProof<SC>,
    ) -> Result<(), ProofMetadataError>
    where
        SC: StarkGenericConfig,
        F: From<Val<SC>>,
        Val<SC>: Copy + PartialEq,
        <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof: FriQueryShapeSource,
    {
        self.matches_shape(&Self::from_proof_with_fri(proof))
    }

    /// Exact structural equality against another shape, including FRI when both set.
    pub fn matches_shape(&self, got: &Self) -> Result<(), ProofMetadataError> {
        if self.ext_degree != got.ext_degree {
            return Err(ProofMetadataError::ExtDegreeMismatch {
                expected: self.ext_degree,
                got: got.ext_degree,
            });
        }
        if self.w_binomial != got.w_binomial {
            return Err(ProofMetadataError::BinomialWMismatch);
        }
        if self.alu_quintic_trinomial != got.alu_quintic_trinomial {
            return Err(ProofMetadataError::QuinticReductionMismatch {
                expected: self.alu_quintic_trinomial,
                got: got.alu_quintic_trinomial,
            });
        }
        if self.alu_variant != got.alu_variant {
            return Err(ProofMetadataError::AluVariantMismatch {
                expected: self.alu_variant,
                got: got.alu_variant,
            });
        }
        if self.rows.as_array() != got.rows.as_array() {
            return Err(ProofMetadataError::RowCountsMismatch {
                expected: self.rows.as_array(),
                got: got.rows.as_array(),
            });
        }
        if self.table_packing != got.table_packing {
            return Err(ProofMetadataError::TablePackingMismatch);
        }
        if self.non_primitives.len() != got.non_primitives.len() {
            return Err(ProofMetadataError::NpoCountMismatch {
                expected: self.non_primitives.len(),
                got: got.non_primitives.len(),
            });
        }
        for (i, (expected, actual)) in self
            .non_primitives
            .iter()
            .zip(&got.non_primitives)
            .enumerate()
        {
            if expected.op_type != actual.op_type {
                return Err(ProofMetadataError::NpoOpTypeMismatch {
                    index: i,
                    expected: expected.op_type.clone(),
                    got: actual.op_type.clone(),
                });
            }
            if expected.air_variant != actual.air_variant {
                return Err(ProofMetadataError::NpoAirVariantMismatch {
                    index: i,
                    expected: expected.air_variant,
                    got: actual.air_variant,
                });
            }
            if expected.public_values_len != actual.public_values_len {
                return Err(ProofMetadataError::NpoPublicValueLenMismatch {
                    index: i,
                    expected: expected.public_values_len,
                    got: actual.public_values_len,
                });
            }
            if expected.rows != actual.rows {
                return Err(ProofMetadataError::NpoRowsMismatch {
                    index: i,
                    expected: expected.rows,
                    got: actual.rows,
                });
            }
            if expected.lanes != actual.lanes {
                return Err(ProofMetadataError::NpoLanesMismatch {
                    index: i,
                    expected: expected.lanes,
                    got: actual.lanes,
                });
            }
        }
        if self.preprocessed != got.preprocessed {
            return Err(ProofMetadataError::PreprocessedMetaMismatch);
        }
        if self.instances.len() != got.instances.len() {
            return Err(ProofMetadataError::OpenedLayoutMismatch {
                index: 0,
                detail: format!(
                    "instance count expected {}, got {}",
                    self.instances.len(),
                    got.instances.len()
                ),
            });
        }
        for (i, (expected, actual)) in self.instances.iter().zip(&got.instances).enumerate() {
            if expected != actual {
                return Err(ProofMetadataError::OpenedLayoutMismatch {
                    index: i,
                    detail: format!("expected {expected:?}, got {actual:?}"),
                });
            }
        }
        if self.has_permutation_commitment != got.has_permutation_commitment {
            return Err(ProofMetadataError::OpenedLayoutMismatch {
                index: 0,
                detail: format!(
                    "permutation commitment expected {}, got {}",
                    self.has_permutation_commitment, got.has_permutation_commitment
                ),
            });
        }
        if self.has_random_commitment != got.has_random_commitment {
            return Err(ProofMetadataError::OpenedLayoutMismatch {
                index: 0,
                detail: format!(
                    "random commitment expected {}, got {}",
                    self.has_random_commitment, got.has_random_commitment
                ),
            });
        }
        match (&self.fri, &got.fri) {
            (None, _) => {}
            (Some(expected), Some(actual)) if expected == actual => {}
            (Some(expected), Some(actual)) => {
                return Err(ProofMetadataError::FriLayoutMismatch(format!(
                    "expected {expected:?}, got {actual:?}"
                )));
            }
            (Some(_), None) => {
                return Err(ProofMetadataError::FriLayoutMismatch(
                    "shape declares a FRI layout but the other side has none; \
                     extract with BatchStarkShape::from_proof_with_fri"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// Return `Ok(())` when `other` fits inside this profile: same structural
    /// flags and every dimension [`Self::union`] can grow is componentwise
    /// `>=` in `self`. Presence growth in the coverage direction (this
    /// profile missing a preprocessed instance, FRI layout, or commitment
    /// that `other` carries) is a structural reject — those fields are not
    /// added by `union`, so accepting them would false-close iteration.
    pub fn covers(&self, other: &Self) -> Result<(), ProofMetadataError> {
        self.assert_same_structure(other)?;
        check_count_fits(
            "const",
            other.rows[PrimitiveTable::Const],
            self.rows[PrimitiveTable::Const],
        )?;
        check_count_fits(
            "public",
            other.rows[PrimitiveTable::Public],
            self.rows[PrimitiveTable::Public],
        )?;
        check_count_fits(
            "alu",
            other.rows[PrimitiveTable::Alu],
            self.rows[PrimitiveTable::Alu],
        )?;
        for (i, (profile, actual)) in self
            .non_primitives
            .iter()
            .zip(&other.non_primitives)
            .enumerate()
        {
            if actual.rows > profile.rows {
                return Err(ProofMetadataError::ProfileOverflow {
                    table: format!("npo[{i}] {:?}", actual.op_type),
                    actual: actual.rows,
                    limit: profile.rows,
                });
            }
        }
        cover_preprocessed(self.preprocessed.as_ref(), other.preprocessed.as_ref())?;
        for (i, (profile, actual)) in self.instances.iter().zip(&other.instances).enumerate() {
            check_count_fits(
                &format!("instances[{i}].degree_bits"),
                actual.degree_bits,
                profile.degree_bits,
            )?;
            if actual.quotient_chunks.len() > profile.quotient_chunks.len() {
                return Err(ProofMetadataError::ProfileOverflow {
                    table: format!("instances[{i}].quotient_chunks.len"),
                    actual: actual.quotient_chunks.len(),
                    limit: profile.quotient_chunks.len(),
                });
            }
            for (j, (limit, got)) in profile
                .quotient_chunks
                .iter()
                .zip(&actual.quotient_chunks)
                .enumerate()
            {
                check_count_fits(
                    &format!("instances[{i}].quotient_chunks[{j}]"),
                    *got,
                    *limit,
                )?;
            }
        }
        match (&self.fri, &other.fri) {
            (None, Some(_)) => {
                return Err(ProofMetadataError::ProfileStructuralMismatch(
                    "fri: profile has none but the other shape carries a layout".into(),
                ));
            }
            (Some(profile), Some(actual)) => {
                check_count_fits(
                    "fri.commit_phase_len",
                    actual.commit_phase_len,
                    profile.commit_phase_len,
                )?;
                check_count_fits(
                    "fri.final_poly_len",
                    actual.final_poly_len,
                    profile.final_poly_len,
                )?;
            }
            (None, None) | (Some(_), None) => {}
        }
        if other.has_permutation_commitment && !self.has_permutation_commitment {
            return Err(ProofMetadataError::ProfileStructuralMismatch(
                "permutation commitment: profile has none but the other shape carries one".into(),
            ));
        }
        if other.has_random_commitment && !self.has_random_commitment {
            return Err(ProofMetadataError::ProfileStructuralMismatch(
                "random commitment: profile has none but the other shape carries one".into(),
            ));
        }
        Ok(())
    }

    /// Componentwise-max union of two shapes that share structure.
    ///
    /// Vector fields (`quotient_chunks`) are extended to the longer length
    /// and then maxed per index — wholesale replacement of a longer `other`
    /// would shrink an overlapping prefix and make the closure recurrence
    /// non-monotone (e.g. `[10] ∪ [5, 1]` must be `[10, 1]`, not `[5, 1]`).
    /// The result covers both inputs.
    ///
    /// Used by [`iterate_profile_closure`] to bump a profile until it covers
    /// `shape(wrapper(profile))`.
    pub fn union(&self, other: &Self) -> Result<Self, ProofMetadataError> {
        self.assert_same_structure(other)?;
        let mut out = self.clone();
        out.rows = self.rows.component_max(other.rows);
        for (dst, src) in out.non_primitives.iter_mut().zip(&other.non_primitives) {
            dst.rows = dst.rows.max(src.rows);
        }
        if let (Some(dst), Some(src)) = (out.preprocessed.as_mut(), other.preprocessed.as_ref()) {
            for (d, s) in dst.instances.iter_mut().zip(&src.instances) {
                if let (Some(d), Some(s)) = (d.as_mut(), s.as_ref()) {
                    d.degree_bits = d.degree_bits.max(s.degree_bits);
                }
            }
        }
        for (d, s) in out.instances.iter_mut().zip(&other.instances) {
            d.degree_bits = d.degree_bits.max(s.degree_bits);
            // Resize, then max. Replacing when `other` is longer drops the
            // overlapping prefix (`[10] ∪ [5, 1]` became `[5, 1]`).
            if s.quotient_chunks.len() > d.quotient_chunks.len() {
                d.quotient_chunks.resize(s.quotient_chunks.len(), 0);
            }
            for (dc, sc) in d.quotient_chunks.iter_mut().zip(&s.quotient_chunks) {
                *dc = (*dc).max(*sc);
            }
        }
        if let (Some(df), Some(sf)) = (out.fri.as_mut(), other.fri.as_ref()) {
            df.commit_phase_len = df.commit_phase_len.max(sf.commit_phase_len);
            df.final_poly_len = df.final_poly_len.max(sf.final_poly_len);
        }
        Ok(out)
    }

    /// Deterministically lift `self` onto `profile`.
    ///
    /// Rejects overflow (any count in `self` strictly larger than `profile`).
    /// Never selects a larger profile from the proof — the returned shape is
    /// exactly `profile`.
    pub fn pad_to_profile(&self, profile: &Self) -> Result<Self, ProofMetadataError> {
        profile.covers(self)?;
        Ok(profile.clone())
    }

    fn assert_same_structure(&self, other: &Self) -> Result<(), ProofMetadataError> {
        if self.ext_degree != other.ext_degree {
            return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                "ext_degree {} vs {}",
                self.ext_degree, other.ext_degree
            )));
        }
        if self.w_binomial != other.w_binomial {
            return Err(ProofMetadataError::ProfileStructuralMismatch(
                "w_binomial".into(),
            ));
        }
        if self.alu_quintic_trinomial != other.alu_quintic_trinomial {
            return Err(ProofMetadataError::ProfileStructuralMismatch(
                "alu_quintic_trinomial".into(),
            ));
        }
        if self.alu_variant != other.alu_variant {
            return Err(ProofMetadataError::ProfileStructuralMismatch(
                "alu_variant".into(),
            ));
        }
        if self.table_packing != other.table_packing {
            return Err(ProofMetadataError::ProfileStructuralMismatch(
                "table_packing".into(),
            ));
        }
        if self.non_primitives.len() != other.non_primitives.len() {
            return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                "npo count {} vs {}",
                self.non_primitives.len(),
                other.non_primitives.len()
            )));
        }
        for (i, (a, b)) in self
            .non_primitives
            .iter()
            .zip(&other.non_primitives)
            .enumerate()
        {
            if a.op_type != b.op_type || a.lanes != b.lanes || a.air_variant != b.air_variant {
                return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                    "npo[{i}] identity"
                )));
            }
            if a.public_values_len != b.public_values_len {
                return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                    "npo[{i}] public_values_len"
                )));
            }
        }
        if self.instances.len() != other.instances.len() {
            return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                "instance count {} vs {}",
                self.instances.len(),
                other.instances.len()
            )));
        }
        for (i, (a, b)) in self.instances.iter().zip(&other.instances).enumerate() {
            if a.trace_local != b.trace_local
                || a.trace_next != b.trace_next
                || a.preprocessed_local != b.preprocessed_local
                || a.preprocessed_next != b.preprocessed_next
                || a.random != b.random
                || a.perm_local != b.perm_local
                || a.perm_next != b.perm_next
                || a.has_lookup_terminal != b.has_lookup_terminal
            {
                return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                    "instances[{i}] opened layout"
                )));
            }
        }
        if let (Some(a), Some(b)) = (&self.preprocessed, &other.preprocessed) {
            if a.matrix_to_instance != b.matrix_to_instance {
                return Err(ProofMetadataError::ProfileStructuralMismatch(
                    "preprocessed matrix_to_instance".into(),
                ));
            }
            if a.instances.len() != b.instances.len() {
                return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                    "preprocessed instance count {} vs {}",
                    a.instances.len(),
                    b.instances.len()
                )));
            }
            for (i, (lhs, rhs)) in a.instances.iter().zip(&b.instances).enumerate() {
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                    if lhs.matrix_index != rhs.matrix_index || lhs.width != rhs.width {
                        return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                            "preprocessed[{i}] identity"
                        )));
                    }
                }
            }
        }
        if let (Some(a), Some(b)) = (&self.fri, &other.fri) {
            if a.commit_pow_count != b.commit_pow_count
                || a.query_count != b.query_count
                || a.log_arities != b.log_arities
                || a.sibling_counts != b.sibling_counts
            {
                return Err(ProofMetadataError::ProfileStructuralMismatch(
                    "fri query identity".into(),
                ));
            }
        }
        Ok(())
    }
}

fn check_count_fits(table: &str, actual: usize, limit: usize) -> Result<(), ProofMetadataError> {
    if actual > limit {
        return Err(ProofMetadataError::ProfileOverflow {
            table: table.into(),
            actual,
            limit,
        });
    }
    Ok(())
}

/// Cover preprocessed instance degrees. Presence growth in the coverage
/// direction is structural: `union` will not introduce a missing instance.
fn cover_preprocessed(
    profile: Option<&PreprocessedShape>,
    other: Option<&PreprocessedShape>,
) -> Result<(), ProofMetadataError> {
    match (profile, other) {
        (None, Some(_)) => Err(ProofMetadataError::ProfileStructuralMismatch(
            "preprocessed: profile has none but the other shape carries metadata".into(),
        )),
        (Some(profile), Some(other)) => {
            if profile.instances.len() != other.instances.len() {
                return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                    "preprocessed instance count {} vs {}",
                    profile.instances.len(),
                    other.instances.len()
                )));
            }
            for (i, (p, a)) in profile.instances.iter().zip(&other.instances).enumerate() {
                match (p, a) {
                    (None, Some(_)) => {
                        return Err(ProofMetadataError::ProfileStructuralMismatch(format!(
                            "preprocessed[{i}]: profile has none but the other shape carries an instance"
                        )));
                    }
                    (Some(p), Some(a)) => {
                        check_count_fits(
                            &format!("preprocessed[{i}].degree_bits"),
                            a.degree_bits,
                            p.degree_bits,
                        )?;
                    }
                    (None, None) | (Some(_), None) => {}
                }
            }
            Ok(())
        }
        (None, None) | (Some(_), None) => Ok(()),
    }
}

/// Pad every registered trace so raw row counts equal `profile`.
///
/// Extra const / public / ALU rows are dummy unused zeros. Extra NPO rows
/// are each table's AIR-valid no-op (Poseidon1/2: sponge `new_start` of the
/// zero state with every CTL flag off; recompose: zero values). Overflow of
/// any table is a hard error. A table that cannot produce a valid dummy row
/// is [`ProofMetadataError::NoValidDummyRow`] — a finding, not a forced pad.
///
/// This only rewrites traces. Call [`pad_preprocessed_to_profile`] (or
/// [`pad_proof_inputs_to_profile`]) so committed multiplicities stay in lockstep;
/// otherwise a padded proof will not verify.
pub fn pad_traces_to_profile<F, S>(
    traces: &mut Traces<F>,
    profile: &BatchStarkShape<S>,
) -> Result<(), ProofMetadataError>
where
    F: Field + PrimeCharacteristicRing,
    S: Copy + PartialEq,
{
    let const_have = traces.const_trace.values.len().max(1);
    let public_have = traces.public_trace.values.len().max(1);
    let alu_have = traces.alu_trace.values.len().max(1);
    check_count_fits("const", const_have, profile.rows[PrimitiveTable::Const])?;
    check_count_fits("public", public_have, profile.rows[PrimitiveTable::Public])?;
    check_count_fits("alu", alu_have, profile.rows[PrimitiveTable::Alu])?;

    pad_const(traces, profile.rows[PrimitiveTable::Const]);
    pad_public(traces, profile.rows[PrimitiveTable::Public]);
    pad_alu(traces, profile.rows[PrimitiveTable::Alu]);

    for entry in &profile.non_primitives {
        let have = traces
            .non_primitive_traces
            .get(&entry.op_type)
            .map_or(0, |t| t.rows());
        check_count_fits(&format!("npo {:?}", entry.op_type), have, entry.rows)?;
        if have == entry.rows {
            continue;
        }
        let Some(trace) = traces.non_primitive_traces.get_mut(&entry.op_type) else {
            return Err(ProofMetadataError::NoValidDummyRow {
                table: format!("npo {:?}", entry.op_type),
                actual: have,
                limit: entry.rows,
            });
        };
        trace.pad_dummy_rows(entry.rows).map_err(|err| match err {
            NpoPadError::NoValidDummyRow => ProofMetadataError::NoValidDummyRow {
                table: format!("npo {:?}", entry.op_type),
                actual: have,
                limit: entry.rows,
            },
        })?;
    }
    Ok(())
}

/// Pad committed preprocessed columns (including multiplicities) to `profile`.
///
/// Dummy prep rows are zeros, so every multiplicity is 0 and the row does not
/// contribute to a lookup. Poseidon1/2 also mark the first dummy row as a
/// chain boundary (second-to-last preprocessed column = 1), matching the AIR's
/// existing power-of-two pad. Call this on the **unpadded** traces so widths
/// can be inferred from `prep_len / have_rows`.
pub fn pad_preprocessed_to_profile<F, PF, S>(
    traces: &Traces<F>,
    primitive_columns: &mut [Vec<PF>],
    non_primitive_columns: &mut NonPrimitivePreprocessedMap<PF>,
    profile: &BatchStarkShape<S>,
) -> Result<(), ProofMetadataError>
where
    PF: PrimeCharacteristicRing,
    S: Copy + PartialEq,
{
    let primitive_have = [
        traces.const_trace.values.len().max(1),
        traces.public_trace.values.len().max(1),
        traces.alu_trace.values.len().max(1),
    ];
    let primitive_targets = [
        profile.rows[PrimitiveTable::Const],
        profile.rows[PrimitiveTable::Public],
        profile.rows[PrimitiveTable::Alu],
    ];
    for (i, (have, target)) in primitive_have.iter().zip(primitive_targets).enumerate() {
        if *have == target {
            continue;
        }
        check_count_fits(&format!("preprocessed primitive[{i}]"), *have, target)?;
        let Some(cols) = primitive_columns.get_mut(i) else {
            return Err(ProofMetadataError::NoValidDummyRow {
                table: format!("preprocessed primitive[{i}]"),
                actual: *have,
                limit: target,
            });
        };
        pad_flat_preprocessed(
            cols,
            *have,
            target,
            None,
            &format!("preprocessed primitive[{i}]"),
        )?;
    }

    for entry in &profile.non_primitives {
        let have = traces
            .non_primitive_traces
            .get(&entry.op_type)
            .map_or(0, |t| t.rows());
        check_count_fits(
            &format!("preprocessed npo {:?}", entry.op_type),
            have,
            entry.rows,
        )?;
        if have == entry.rows {
            continue;
        }
        let Some(cols) = non_primitive_columns.get_mut(&entry.op_type) else {
            return Err(ProofMetadataError::NoValidDummyRow {
                table: format!("preprocessed npo {:?}", entry.op_type),
                actual: have,
                limit: entry.rows,
            });
        };
        let chain_boundary = poseidon_perm_needs_chain_boundary(&entry.op_type);
        pad_flat_preprocessed(
            cols,
            have,
            entry.rows,
            chain_boundary.then_some(mark_poseidon_first_dummy_chain_boundary),
            &format!("preprocessed npo {:?}", entry.op_type),
        )?;
    }
    Ok(())
}

/// Pad traces and committed preprocessed columns together.
///
/// Preprocessed is padded first so dummy-row widths are inferred from the
/// unpadded counts. Prefer [`CircuitProverData::pad_to_profile`], which also
/// marks setup commitments stale so prove rebuilds `ProverData` even when
/// the padded counts stay inside the same power-of-two height.
pub fn pad_proof_inputs_to_profile<F, PF, S>(
    traces: &mut Traces<F>,
    primitive_columns: &mut [Vec<PF>],
    non_primitive_columns: &mut NonPrimitivePreprocessedMap<PF>,
    profile: &BatchStarkShape<S>,
) -> Result<(), ProofMetadataError>
where
    F: Field + PrimeCharacteristicRing,
    PF: PrimeCharacteristicRing,
    S: Copy + PartialEq,
{
    pad_preprocessed_to_profile(traces, primitive_columns, non_primitive_columns, profile)?;
    pad_traces_to_profile(traces, profile)
}

impl<SC: StarkGenericConfig> CircuitProverData<SC> {
    /// Pad traces and this circuit's committed preprocessed columns to `profile`.
    ///
    /// Marks [`Self::preprocessed_stale`] so the next prove rebuilds
    /// `ProverData` even when dummy rows fit in the existing power-of-two
    /// domain. Also drops the ALU schedule cache (op count changed).
    pub fn pad_to_profile<F, S>(
        &mut self,
        traces: &mut Traces<F>,
        profile: &BatchStarkShape<S>,
    ) -> Result<(), ProofMetadataError>
    where
        F: Field + PrimeCharacteristicRing,
        S: Copy + PartialEq,
    {
        pad_proof_inputs_to_profile(
            traces,
            &mut self.primitive_columns,
            &mut self.non_primitive_columns,
            profile,
        )?;
        self.preprocessed_stale = true;
        *self.alu_schedule_cache.borrow_mut() = None;
        Ok(())
    }
}

fn poseidon_perm_needs_chain_boundary(op_type: &NpoTypeId) -> bool {
    let id = op_type.as_str();
    id.starts_with("poseidon2_perm/") || id.starts_with("poseidon1_perm/")
}

fn mark_poseidon_first_dummy_chain_boundary<PF: PrimeCharacteristicRing>(row: &mut [PF]) {
    // AIR `preprocessed_trace`: first pad row has chain-start at width-2.
    if row.len() >= 2 {
        row[row.len() - 2] = PF::ONE;
    }
}

fn pad_flat_preprocessed<PF: PrimeCharacteristicRing>(
    cols: &mut Vec<PF>,
    have: usize,
    target: usize,
    first_dummy: Option<fn(&mut [PF])>,
    table: &str,
) -> Result<(), ProofMetadataError> {
    if have == target {
        return Ok(());
    }
    if have == 0 || !cols.len().is_multiple_of(have) {
        return Err(ProofMetadataError::NoValidDummyRow {
            table: table.into(),
            actual: have,
            limit: target,
        });
    }
    let width = cols.len() / have;
    if width == 0 {
        return Err(ProofMetadataError::NoValidDummyRow {
            table: table.into(),
            actual: have,
            limit: target,
        });
    }
    cols.resize(width * target, PF::ZERO);
    if let Some(mark) = first_dummy {
        mark(&mut cols[have * width..(have + 1) * width]);
    }
    Ok(())
}

fn pad_const<F: PrimeCharacteristicRing>(traces: &mut Traces<F>, target: usize) {
    while traces.const_trace.values.len() < target {
        traces.const_trace.index.push(WitnessId(0));
        traces.const_trace.values.push(F::ZERO);
    }
}

fn pad_public<F: PrimeCharacteristicRing>(traces: &mut Traces<F>, target: usize) {
    while traces.public_trace.values.len() < target {
        traces.public_trace.index.push(WitnessId(0));
        traces.public_trace.values.push(F::ZERO);
    }
}

fn pad_alu<F: Field>(traces: &mut Traces<F>, target: usize) {
    use p3_circuit::ops::AluOpKind;
    while traces.alu_trace.values.len() < target {
        traces.alu_trace.op_kind.push(AluOpKind::Add);
        traces
            .alu_trace
            .values
            .push([F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
        traces
            .alu_trace
            .indices
            .push([WitnessId(0), WitnessId(0), WitnessId(0), WitnessId(0)]);
    }
}

/// Error from [`iterate_profile_closure`].
#[derive(Debug)]
pub enum ProfileClosureError<E> {
    /// The wrapper-shape oracle failed.
    Measure(E),
    /// A [`ProofMetadataError`] raised by `covers` / `union`.
    Shape(ProofMetadataError),
}

impl<E: core::fmt::Display> core::fmt::Display for ProfileClosureError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Measure(e) => write!(f, "profile-closure measure failed: {e}"),
            Self::Shape(e) => write!(f, "profile-closure shape error: {e}"),
        }
    }
}

impl<E: core::fmt::Debug + core::fmt::Display> core::error::Error for ProfileClosureError<E> {}

/// Iterate the value-free shape recurrence until a closed profile is found.
///
/// `measure_wrapper(profile)` must return `shape(wrapper(profile))` for every
/// semantic class the caller cares about (or the componentwise-max of those
/// classes). Closure is `profile.covers(wrapper)`. Every dimension `union`
/// can grow — primitive/NPO rows, preprocessed instance degrees, opened
/// `degree_bits` / `quotient_chunks`, FRI phase/final lengths — is covered
/// componentwise; presence growth in the coverage direction is structural.
/// Counts are bumped via [`BatchStarkShape::union`]; a structural mismatch
/// kills the construction.
///
/// This bootstrap depends only on shapes, never on proof or verification-key
/// values.
pub fn iterate_profile_closure<F, E>(
    seed: BatchStarkShape<F>,
    mut measure_wrapper: impl FnMut(&BatchStarkShape<F>) -> Result<BatchStarkShape<F>, E>,
    max_iters: usize,
) -> Result<BatchStarkShape<F>, ProfileClosureError<E>>
where
    F: Copy + PartialEq,
{
    let mut profile = seed;
    for _ in 0..max_iters {
        let wrapper = measure_wrapper(&profile).map_err(ProfileClosureError::Measure)?;
        match profile.covers(&wrapper) {
            Ok(()) => return Ok(profile),
            Err(ProofMetadataError::ProfileOverflow { .. }) => {
                profile = profile
                    .union(&wrapper)
                    .map_err(ProfileClosureError::Shape)?;
            }
            Err(e) => return Err(ProfileClosureError::Shape(e)),
        }
    }
    Err(ProfileClosureError::Shape(
        ProofMetadataError::ProfileDidNotClose { iters: max_iters },
    ))
}

/// Convenience alias used by tests that do not care about the measure error.
pub type InfalliblyClosed<F> =
    Result<BatchStarkShape<F>, ProfileClosureError<core::convert::Infallible>>;
