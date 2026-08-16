//! Acceptance tests for const-value binding (finding F1).
//!
//! The global preprocessed commitment is the circuit's verification key: it must
//! authenticate `Op::Const` values, not just their witness indices. Two circuits
//! that are identical except for one constant's *value* must have different
//! preprocessed commitments, and a proof for the altered circuit must not verify
//! against the canonical circuit's common data.
//!
//! Both tests fail on the pre-fix layout (const preprocessed = `[ext_mult, index]`
//! with the value only in the prover-chosen main trace) and pass once the value
//! coefficients are committed and the witness-bus send reads them from the
//! preprocessed columns.

use p3_batch_stark::common::GlobalPreprocessed;
use p3_batch_stark::{CommonData, ProverData};
use p3_circuit::CircuitBuilder;
use p3_circuit::tables::Traces;
use p3_circuit_prover::common::{NpoAirBuilder, NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::config::KoalaBearConfig;
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, TablePacking, config,
};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
const D: usize = 4;

/// The canonical constant and an altered value; both fresh (distinct from the
/// builder's deduplicated 0/1 constants) so the two circuits share one topology,
/// witness layout, and row profile, differing only in the constant's value.
const CANONICAL_CONST: u64 = 123_456_789;
const ALTERED_CONST: u64 = 987_654_321;

/// Everything needed to prove and verify one instance of the test circuit.
struct Harness {
    prover: BatchStarkProver<KoalaBearConfig>,
    circuit_prover_data: CircuitProverData<KoalaBearConfig>,
    traces: Traces<EF>,
    /// Field-wise copy of the setup's `CommonData` (the canonical verification data).
    canonical_common: CommonData<KoalaBearConfig>,
}

/// Clone the preprocessed binding of a [`CommonData`]. Lookups are intentionally
/// empty: the verifier always rebuilds them from the reconstructed AIRs.
fn clone_common(common: &CommonData<KoalaBearConfig>) -> CommonData<KoalaBearConfig> {
    CommonData::new(
        common.preprocessed.as_ref().map(|gp| GlobalPreprocessed {
            commitment: gp.commitment.clone(),
            instances: gp.instances.clone(),
            matrix_to_instance: gp.matrix_to_instance.clone(),
        }),
        Vec::new(),
    )
}

/// Build and set up the test circuit `assert_zero(const(c) - public_input)` with
/// the public input satisfying it (`p = c`).
fn setup(c: u64) -> Harness {
    let mut builder = CircuitBuilder::<EF>::new();
    let k = builder.alloc_const(EF::from_u64(c), "f1_const");
    let p = builder.alloc_public_input("f1_public");
    let diff = builder.sub(k, p);
    builder.assert_zero(diff);
    let circuit = builder.build().expect("circuit build");

    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[EF::from_u64(c)])
        .expect("set public inputs");
    let traces = runner.run().expect("run circuit");

    let table_packing = TablePacking::new(1, 1);
    let stark_config = config::koala_bear();
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![];
    let air_builders: Vec<Box<dyn NpoAirBuilder<KoalaBearConfig, D>>> = vec![];
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .expect("derive airs and preprocessed columns");
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let prover_data_stark = ProverData::from_airs_and_degrees(&stark_config, &airs, &degrees);
    let canonical_common = clone_common(&prover_data_stark.common);
    let circuit_prover_data =
        CircuitProverData::new(prover_data_stark, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(stark_config).with_table_packing(table_packing);

    Harness {
        prover,
        circuit_prover_data,
        traces,
        canonical_common,
    }
}

/// Two circuits identical except for one `Op::Const` *value* must not share a
/// global preprocessed commitment: the commitment is the verification key, and a
/// vk that does not authenticate constants authenticates the wrong circuit.
#[test]
fn f1_const_value_binding_is_required() {
    let canonical = setup(CANONICAL_CONST);
    let altered = setup(ALTERED_CONST);

    let commitment = |h: &Harness| {
        h.canonical_common
            .preprocessed
            .as_ref()
            .expect("primitive tables always commit preprocessed columns")
            .commitment
            .clone()
    };

    assert_ne!(
        commitment(&canonical),
        commitment(&altered),
        "circuits differing only in an Op::Const value must have different \
         global preprocessed commitments"
    );
}

/// A proof for the altered-const circuit must be rejected (not accepted, not a
/// panic) when verified against the canonical circuit's common data.
#[test]
fn f1_altered_const_proof_rejected_against_canonical_common() {
    let canonical = setup(CANONICAL_CONST);
    let altered = setup(ALTERED_CONST);

    let mut proof = altered
        .prover
        .prove_all_tables(&altered.traces, &altered.circuit_prover_data)
        .expect("prove altered circuit");

    // Sanity: the altered proof verifies against its own common data.
    altered
        .prover
        .verify_all_tables::<EF>(&proof)
        .expect("altered proof verifies against its own common data");

    // A verifier that pins the canonical circuit's common data must reject the
    // altered-const proof.
    proof.stark_common = canonical.canonical_common;
    let result = altered.prover.verify_all_tables::<EF>(&proof);
    assert!(
        result.is_err(),
        "altered-const proof must not verify against the canonical circuit's \
         common data"
    );
}
