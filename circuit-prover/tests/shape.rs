//! Value-free `BatchStarkShape` extraction, exact matching, pad-to-profile,
//! and profile-closure tests.

use core::any::Any;
use p3_baby_bear::BabyBear;
use p3_batch_stark::ProverData;
use p3_circuit::builder::CircuitBuilder;
use p3_circuit_prover::ConstraintProfile;
use p3_circuit_prover::batch_stark_prover::{
    AirVariant, BatchStarkProver, CircuitProverData, PrimitiveTable, ProofMetadataError, RowCounts,
    TablePacking,
};
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::config::{self, BabyBearConfig};

use p3_circuit::ops::{
    NpoTypeId, Poseidon2CircuitRow, Poseidon2Config, Poseidon2Trace, RecomposeCircuitRow,
    RecomposeTrace, RecomposeTraceKind,
};
use p3_circuit::tables::{NonPrimitiveTrace, NpoPadError};
use p3_circuit::types::WitnessId;
use p3_circuit_prover::shape::{
    BatchStarkShape, FriQueryShape, NpoShapeEntry, iterate_profile_closure,
    pad_preprocessed_to_profile, pad_traces_to_profile,
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

fn baby_bear_base_proof() -> p3_circuit_prover::batch_stark_prover::BatchStarkProof<BabyBearConfig>
{
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.add(x, y);
    let c = builder.define_const(BabyBear::from_u64(3));
    let diff = builder.sub(z, c);
    builder.assert_zero(diff);
    let circuit = builder.build().unwrap();

    let cfg = config::baby_bear();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[BabyBear::from_u64(1), BabyBear::from_u64(2)])
        .unwrap();
    let traces = runner.run().unwrap();
    BatchStarkProver::new(cfg)
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap()
}

fn baby_bear_circuit_and_traces() -> (
    p3_circuit::Circuit<BabyBear>,
    p3_circuit::tables::Traces<BabyBear>,
    CircuitProverData<BabyBearConfig>,
    BabyBearConfig,
) {
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.add(x, y);
    let c = builder.define_const(BabyBear::from_u64(3));
    let diff = builder.sub(z, c);
    builder.assert_zero(diff);
    let circuit = builder.build().unwrap();

    let cfg = config::baby_bear();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[BabyBear::from_u64(1), BabyBear::from_u64(2)])
        .unwrap();
    let traces = runner.run().unwrap();
    (circuit, traces, circuit_prover_data, cfg)
}

#[test]
fn shape_round_trips_from_proof() {
    let proof = baby_bear_base_proof();
    let shape = BatchStarkShape::<BabyBear>::from_proof(&proof);
    assert_eq!(shape.matches_proof(&proof), Ok(()));
    assert_eq!(shape.ext_degree, 1);
    assert_eq!(shape.alu_variant, AirVariant::Optimized);
    assert!(shape.non_primitives.is_empty());
    assert!(shape.fri.is_none());
    assert!(!shape.instances.is_empty());
}

#[test]
fn shape_with_fri_layout_round_trips() {
    let proof = baby_bear_base_proof();
    let fri = FriQueryShape::from_fri_proof(&proof.proof.opening_proof);
    assert!(fri.query_count > 0);
    assert!(fri.commit_phase_len > 0);
    let shape = BatchStarkShape::<BabyBear>::from_proof_with_fri(&proof);
    assert_eq!(shape.fri.as_ref(), Some(&fri));
    assert_eq!(shape.matches_proof_with_fri(&proof), Ok(()));
    // Metadata-only matches_proof cannot see FRI; a locked descriptor must
    // use the FRI-aware path or it reports the missing layout.
    assert!(matches!(
        shape.matches_proof(&proof),
        Err(ProofMetadataError::FriLayoutMismatch(_))
    ));
    let got = BatchStarkShape::<BabyBear>::from_proof(&proof).with_fri(fri);
    assert_eq!(shape.matches_shape(&got), Ok(()));
    let mut bad_fri = got.clone();
    if let Some(f) = bad_fri.fri.as_mut() {
        f.query_count += 1;
    }
    assert!(matches!(
        shape.matches_shape(&bad_fri),
        Err(ProofMetadataError::FriLayoutMismatch(_))
    ));
}

#[test]
fn shape_rejects_row_count_mutation() {
    let proof = baby_bear_base_proof();
    let mut shape = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let mut rows = shape.rows.as_array();
    rows[0] += 8;
    shape.rows = RowCounts::new(rows);
    assert!(matches!(
        shape.matches_proof(&proof),
        Err(ProofMetadataError::RowCountsMismatch { .. })
    ));
}

#[test]
fn shape_rejects_packing_mutation() {
    let proof = baby_bear_base_proof();
    let mut shape = BatchStarkShape::<BabyBear>::from_proof(&proof);
    shape.table_packing = TablePacking::new(4, 4);
    assert_eq!(
        shape.matches_proof(&proof),
        Err(ProofMetadataError::TablePackingMismatch)
    );
}

#[test]
fn native_verify_matching_shape_accepts_and_rejects() {
    let proof = baby_bear_base_proof();
    let shape = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let cfg = config::baby_bear();
    let prover = BatchStarkProver::new(cfg);
    prover
        .verify_all_tables_matching_shape::<BabyBear>(&proof, &shape)
        .unwrap();

    let mut bad = shape.clone();
    bad.alu_variant = AirVariant::Baseline;
    let err = prover
        .verify_all_tables_matching_shape::<BabyBear>(&proof, &bad)
        .unwrap_err();
    match err {
        p3_circuit_prover::batch_stark_prover::BatchStarkProverError::InvalidMetadata(
            ProofMetadataError::AluVariantMismatch { .. },
        ) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn pad_to_profile_rejects_overflow_and_returns_exact_profile() {
    let proof = baby_bear_base_proof();
    let actual = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let mut profile = actual.clone();
    let mut rows = profile.rows.as_array();
    rows[PrimitiveTable::Const as usize] += 16;
    profile.rows = RowCounts::new(rows);

    let padded = actual.pad_to_profile(&profile).unwrap();
    assert_eq!(padded.rows.as_array(), profile.rows.as_array());

    let overflow = profile.pad_to_profile(&actual);
    assert!(matches!(
        overflow,
        Err(ProofMetadataError::ProfileOverflow { .. })
    ));
}

#[test]
fn pad_traces_to_profile_extends_primitive_rows() {
    let (_circuit, mut traces, _data, _cfg) = baby_bear_circuit_and_traces();
    let proof = baby_bear_base_proof();
    let mut profile = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let mut rows = profile.rows.as_array();
    rows[PrimitiveTable::Const as usize] += 4;
    rows[PrimitiveTable::Alu as usize] += 4;
    profile.rows = RowCounts::new(rows);

    pad_traces_to_profile(&mut traces, &profile).unwrap();
    assert_eq!(
        traces.const_trace.values.len(),
        profile.rows[PrimitiveTable::Const]
    );
    assert_eq!(
        traces.alu_trace.values.len(),
        profile.rows[PrimitiveTable::Alu]
    );

    let mut too_small = profile.clone();
    too_small.rows = RowCounts::new([1, 1, 1]);
    assert!(matches!(
        pad_traces_to_profile(&mut traces, &too_small),
        Err(ProofMetadataError::ProfileOverflow { .. })
    ));
}

#[test]
fn iterate_profile_closure_reaches_fixed_point() {
    let proof = baby_bear_base_proof();
    let seed = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let target_const = seed.rows[PrimitiveTable::Const] + 6;

    let mut calls = 0usize;
    let closed = iterate_profile_closure(
        seed.clone(),
        |profile| {
            calls += 1;
            let mut wrapper = profile.clone();
            let have = wrapper.rows[PrimitiveTable::Const];
            if have < target_const {
                let mut rows = wrapper.rows.as_array();
                rows[0] = have + 2;
                wrapper.rows = RowCounts::new(rows);
            }
            Ok::<_, core::convert::Infallible>(wrapper)
        },
        16,
    )
    .unwrap();

    assert!(calls > 1);
    assert_eq!(closed.rows[PrimitiveTable::Const], target_const);
    assert!(closed.covers(&seed).is_ok());
}

#[test]
fn iterate_profile_closure_stops_on_structural_mismatch() {
    let proof = baby_bear_base_proof();
    let seed = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let err = iterate_profile_closure(
        seed,
        |profile| {
            let mut wrapper = profile.clone();
            wrapper.ext_degree = 4;
            Ok::<_, core::convert::Infallible>(wrapper)
        },
        4,
    )
    .unwrap_err();
    match err {
        p3_circuit_prover::shape::ProfileClosureError::Shape(
            ProofMetadataError::ProfileStructuralMismatch(_),
        ) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

/// Grow only the verifier dimensions `union` mutates outside rows/NPO rows.
fn grow_instance_preprocessed_fri(shape: &mut BatchStarkShape<BabyBear>, delta: usize) {
    if let Some(prep) = shape.preprocessed.as_mut() {
        for inst in prep.instances.iter_mut().flatten() {
            inst.degree_bits += delta;
        }
    }
    for inst in &mut shape.instances {
        inst.degree_bits += delta;
        if inst.quotient_chunks.is_empty() {
            inst.quotient_chunks.push(delta);
        } else {
            inst.quotient_chunks[0] += delta;
        }
    }
    if let Some(fri) = shape.fri.as_mut() {
        fri.commit_phase_len += delta;
        fri.final_poly_len += delta;
    }
}

#[test]
fn covers_rejects_instance_preprocessed_fri_growth() {
    let proof = baby_bear_base_proof();
    let seed = BatchStarkShape::<BabyBear>::from_proof_with_fri(&proof);
    assert!(seed.preprocessed.is_some());
    assert!(seed.fri.is_some());
    assert!(!seed.instances.is_empty());

    let mut grown = seed.clone();
    grow_instance_preprocessed_fri(&mut grown, 1);
    // The hole at 7800909e: covers() ignored these fields and returned Ok.
    assert!(matches!(
        seed.covers(&grown),
        Err(ProofMetadataError::ProfileOverflow { .. })
    ));
    assert_eq!(seed.covers(&seed), Ok(()));
}

#[test]
fn iterate_profile_closure_bumps_instance_preprocessed_fri_dims() {
    let proof = baby_bear_base_proof();
    let seed = BatchStarkShape::<BabyBear>::from_proof_with_fri(&proof);
    let mut target = seed.clone();
    grow_instance_preprocessed_fri(&mut target, 2);

    let mut calls = 0usize;
    let closed = iterate_profile_closure(
        seed.clone(),
        |_| {
            calls += 1;
            Ok::<_, core::convert::Infallible>(target.clone())
        },
        8,
    )
    .unwrap();

    assert!(
        calls > 1,
        "closure must iterate a union bump, not accept the seed"
    );
    assert_eq!(
        closed.instances[0].degree_bits,
        target.instances[0].degree_bits
    );
    assert_eq!(
        closed.instances[0].quotient_chunks,
        target.instances[0].quotient_chunks
    );
    assert_eq!(
        closed.preprocessed.as_ref().and_then(|p| {
            p.instances
                .iter()
                .find_map(|inst| inst.as_ref().map(|m| m.degree_bits))
        }),
        target.preprocessed.as_ref().and_then(|p| {
            p.instances
                .iter()
                .find_map(|inst| inst.as_ref().map(|m| m.degree_bits))
        })
    );
    assert_eq!(
        closed
            .fri
            .as_ref()
            .map(|f| (f.commit_phase_len, f.final_poly_len)),
        target
            .fri
            .as_ref()
            .map(|f| (f.commit_phase_len, f.final_poly_len))
    );
    assert_ne!(closed, seed, "must not return the stale profile");
}

#[test]
fn covers_rejects_fri_presence_growth() {
    let proof = baby_bear_base_proof();
    let without = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let with = BatchStarkShape::<BabyBear>::from_proof_with_fri(&proof);
    assert!(without.fri.is_none());
    assert!(with.fri.is_some());

    assert!(matches!(
        without.covers(&with),
        Err(ProofMetadataError::ProfileStructuralMismatch(_))
    ));
    // A FRI-locked profile may still cover a FRI-less actual (pad-to-profile).
    assert_eq!(with.covers(&without), Ok(()));

    let err = iterate_profile_closure(
        without.clone(),
        |_| Ok::<_, core::convert::Infallible>(with.clone()),
        4,
    )
    .unwrap_err();
    match err {
        p3_circuit_prover::shape::ProfileClosureError::Shape(
            ProofMetadataError::ProfileStructuralMismatch(_),
        ) => {}
        other => panic!("FRI-less profile must reject, not close: {other:?}"),
    }
}

/// Sol's counterexample at 1700dd2: wholesale replace when `other` is longer
/// turned `[10] ∪ [5, 1]` into `[5, 1]`, shrinking the established prefix.
#[test]
fn union_quotient_chunks_is_componentwise_max_on_mixed_length() {
    let proof = baby_bear_base_proof();
    let base = BatchStarkShape::<BabyBear>::from_proof(&proof);
    assert!(!base.instances.is_empty());

    let mut short_high = base.clone();
    let mut long_low = base.clone();
    short_high.instances[0].quotient_chunks = vec![10];
    long_low.instances[0].quotient_chunks = vec![5, 1];

    let united = short_high.union(&long_low).unwrap();
    assert_eq!(united.instances[0].quotient_chunks, vec![10, 1]);
    assert_eq!(united.covers(&short_high), Ok(()));
    assert_eq!(united.covers(&long_low), Ok(()));

    // Symmetric: dest already longer, overlapping prefix still maxes.
    let united_rev = long_low.union(&short_high).unwrap();
    assert_eq!(united_rev.instances[0].quotient_chunks, vec![10, 1]);
    assert_eq!(united_rev.covers(&short_high), Ok(()));
    assert_eq!(united_rev.covers(&long_low), Ok(()));

    // Mixed length *and* mixed values on both sides.
    let mut a = base.clone();
    let mut b = base.clone();
    a.instances[0].quotient_chunks = vec![10, 3];
    b.instances[0].quotient_chunks = vec![8, 1, 2];
    let mixed = a.union(&b).unwrap();
    assert_eq!(mixed.instances[0].quotient_chunks, vec![10, 3, 2]);
    assert_eq!(mixed.covers(&a), Ok(()));
    assert_eq!(mixed.covers(&b), Ok(()));
}

/// Closure recurrence must be monotone: each iterate covers its predecessor.
/// With the 1700dd2 replace, seed `[10]` ∪ wrapper `[5, 1]` closed as `[5, 1]`
/// and the next profile no longer covered the seed.
#[test]
fn iterate_profile_closure_never_shrinks_profile() {
    let proof = baby_bear_base_proof();
    let mut seed = BatchStarkShape::<BabyBear>::from_proof(&proof);
    seed.instances[0].quotient_chunks = vec![10];

    let mut wrapper_shape = seed.clone();
    wrapper_shape.instances[0].quotient_chunks = vec![5, 1];

    let mut prev = seed.clone();
    let mut steps = 0usize;
    let closed = iterate_profile_closure(
        seed.clone(),
        |profile| {
            steps += 1;
            assert_eq!(
                profile.covers(&prev),
                Ok(()),
                "profile shrank across closure iterations at step {steps}"
            );
            prev = profile.clone();
            Ok::<_, core::convert::Infallible>(wrapper_shape.clone())
        },
        8,
    )
    .unwrap();

    assert!(steps > 1, "must union-bump, not accept the seed");
    assert_eq!(closed.instances[0].quotient_chunks, vec![10, 1]);
    assert_eq!(closed.covers(&seed), Ok(()));
    assert_eq!(closed.covers(&wrapper_shape), Ok(()));
    assert_eq!(closed.covers(&prev), Ok(()));
}

fn poseidon2_dummy_row() -> Poseidon2CircuitRow<BabyBear> {
    Poseidon2CircuitRow {
        new_start: true,
        merkle_path: false,
        mmcs_bit: false,
        mmcs_bit2: false,
        mmcs_index_sum: BabyBear::ZERO,
        input_values: vec![BabyBear::ZERO; 16],
        in_ctl: vec![false; 16],
        input_indices: vec![0; 16],
        out_ctl: vec![false; 8],
        output_indices: vec![0; 8],
        mmcs_index_sum_idx: 0,
        mmcs_ctl_enabled: false,
    }
}

#[test]
fn pad_traces_to_profile_extends_npo_poseidon2_and_recompose() {
    let (_circuit, mut traces, _data, _cfg) = baby_bear_circuit_and_traces();
    let proof = baby_bear_base_proof();
    let mut profile = BatchStarkShape::<BabyBear>::from_proof(&proof);

    let p2 = NpoTypeId::poseidon2_perm(Poseidon2Config::BABY_BEAR_D1_W16);
    let rec = NpoTypeId::recompose();
    traces.non_primitive_traces.insert(
        p2.clone(),
        Box::new(Poseidon2Trace {
            op_type: p2.clone(),
            operations: vec![poseidon2_dummy_row(), poseidon2_dummy_row()],
        }),
    );
    traces.non_primitive_traces.insert(
        rec.clone(),
        Box::new(RecomposeTrace {
            operations: vec![RecomposeCircuitRow {
                input_wids: vec![WitnessId(0); 4],
                output_wid: WitnessId(1),
                values: vec![BabyBear::ONE; 4],
            }],
            kind: RecomposeTraceKind::Standard,
        }),
    );
    profile.non_primitives = vec![
        NpoShapeEntry {
            op_type: p2.clone(),
            rows: 8,
            lanes: 1,
            public_values_len: 0,
            air_variant: AirVariant::Baseline,
        },
        NpoShapeEntry {
            op_type: rec.clone(),
            rows: 5,
            lanes: 1,
            public_values_len: 0,
            air_variant: AirVariant::Baseline,
        },
    ];

    pad_traces_to_profile(&mut traces, &profile).unwrap();
    assert_eq!(traces.non_primitive_traces.get(&p2).unwrap().rows(), 8);
    assert_eq!(traces.non_primitive_traces.get(&rec).unwrap().rows(), 5);

    let p2_trace = traces
        .non_primitive_trace::<Poseidon2Trace<BabyBear>>(&p2)
        .unwrap();
    assert!(p2_trace.operations[2..].iter().all(|row| {
        row.new_start && !row.merkle_path && row.input_values.iter().all(|v| *v == BabyBear::ZERO)
    }));
}

#[test]
fn pad_traces_to_profile_unknown_npo_is_a_finding() {
    struct UnknownTrace;

    impl NonPrimitiveTrace<BabyBear> for UnknownTrace {
        fn op_type(&self) -> NpoTypeId {
            NpoTypeId::new("statement")
        }
        fn rows(&self) -> usize {
            1
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn boxed_clone(&self) -> Box<dyn NonPrimitiveTrace<BabyBear>> {
            Box::new(UnknownTrace)
        }
    }

    let (_circuit, mut traces, _data, _cfg) = baby_bear_circuit_and_traces();
    let proof = baby_bear_base_proof();
    let mut profile = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let op = NpoTypeId::new("statement");
    traces
        .non_primitive_traces
        .insert(op.clone(), Box::new(UnknownTrace));
    profile.non_primitives = vec![NpoShapeEntry {
        op_type: op,
        rows: 4,
        lanes: 1,
        public_values_len: 0,
        air_variant: AirVariant::Baseline,
    }];

    let err = pad_traces_to_profile(&mut traces, &profile).unwrap_err();
    assert!(
        matches!(err, ProofMetadataError::NoValidDummyRow { .. }),
        "unknown NPO must be a finding, not a forced pad: {err:?}"
    );
    let _ = NpoPadError::NoValidDummyRow;
}

#[test]
fn pad_preprocessed_to_profile_marks_poseidon_chain_boundary() {
    let (_circuit, traces, _data, _cfg) = baby_bear_circuit_and_traces();
    let proof = baby_bear_base_proof();
    let mut profile = BatchStarkShape::<BabyBear>::from_proof(&proof);
    let p2 = NpoTypeId::poseidon2_perm(Poseidon2Config::BABY_BEAR_D1_W16);
    let mut traces = traces;
    traces.non_primitive_traces.insert(
        p2.clone(),
        Box::new(Poseidon2Trace {
            op_type: p2.clone(),
            operations: vec![poseidon2_dummy_row(), poseidon2_dummy_row()],
        }),
    );
    profile.non_primitives = vec![NpoShapeEntry {
        op_type: p2.clone(),
        rows: 6,
        lanes: 1,
        public_values_len: 0,
        air_variant: AirVariant::Baseline,
    }];

    let width = 10usize;
    let mut primitive = vec![Vec::new(), Vec::new(), Vec::new()];
    let mut npo = p3_circuit::ops::NonPrimitivePreprocessedMap::default();
    npo.insert(p2.clone(), vec![BabyBear::from_u64(7); 2 * width]);

    pad_preprocessed_to_profile(&traces, &mut primitive, &mut npo, &profile).unwrap();
    let padded = npo.get(&p2).unwrap();
    assert_eq!(padded.len(), 6 * width);
    assert_eq!(padded[2 * width + width - 2], BabyBear::ONE);
    assert!(padded[3 * width..].iter().all(|v| *v == BabyBear::ZERO));
}

/// Mint-shaped (few NPO rows) padded to a larger transfer-like profile:
/// re-extracted shape matches the frozen FRI-locked profile and the padded
/// proof verifies.
#[test]
fn pad_mint_shaped_traces_to_transfer_like_profile_verifies() {
    use p3_circuit::ops::poseidon2_perm::Poseidon2PermCallBase;
    use p3_circuit::ops::{KoalaBearD1Width16, generate_poseidon2_trace};
    use p3_circuit_prover::batch_stark_prover::{
        Poseidon2Preprocessor, poseidon2_air_builders_d5, poseidon2_table_provers_d5,
    };
    use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
    use p3_circuit_prover::config::KoalaBearConfig;
    use p3_field::extension::QuinticTrinomialExtensionField;
    use p3_koala_bear::{KoalaBear, default_koalabear_poseidon2_16};
    use p3_symmetric::Permutation;
    use p3_test_utils::LiftPermToQuintic;

    const D: usize = 5;
    type EF5 = QuinticTrinomialExtensionField<KoalaBear>;

    let inner_perm = default_koalabear_poseidon2_16();
    let mut sponge0 = [KoalaBear::ZERO; 16];
    sponge0[0] = KoalaBear::from_u64(11);
    sponge0[1] = KoalaBear::from_u64(13);
    let sponge_out = inner_perm.permute(sponge0);
    let lift_perm = LiftPermToQuintic::new(inner_perm);

    let in0 = EF5::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(11),
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ])
    .unwrap();
    let in1 = EF5::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(13),
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ])
    .unwrap();
    let exp0 = EF5::from_basis_coefficients_slice(&[
        sponge_out[0],
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ])
    .unwrap();
    let exp1 = EF5::from_basis_coefficients_slice(&[
        sponge_out[1],
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ])
    .unwrap();

    let mut builder = CircuitBuilder::<EF5>::new();
    builder.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
        generate_poseidon2_trace::<EF5, KoalaBearD1Width16>,
        lift_perm,
    );
    let in_a = builder.public_input();
    let in_b = builder.public_input();
    let mut perm_inputs: [Option<_>; 16] = [None; 16];
    perm_inputs[0] = Some(in_a);
    perm_inputs[1] = Some(in_b);
    let (_pid, hash_outputs) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KOALA_BEAR_D1_W16,
            new_start: true,
            inputs: perm_inputs,
            out_ctl: [true; 8],
            return_all_outputs: false,
            absorb_len: 0,
        })
        .unwrap();
    let e0 = builder.public_input();
    let e1 = builder.public_input();
    let h0_diff = builder.sub(hash_outputs[0].unwrap(), e0);
    let h1_diff = builder.sub(hash_outputs[1].unwrap(), e1);
    builder.assert_zero(h0_diff);
    builder.assert_zero(h1_diff);
    let circuit = builder.build().unwrap();

    let cfg = p3_circuit_prover::config::koala_bear();
    let npo_prep: Vec<Box<dyn NpoPreprocessor<KoalaBear>>> = vec![Box::new(Poseidon2Preprocessor)];
    let air_builders = poseidon2_air_builders_d5::<KoalaBearConfig>();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[in0, in1, exp0, exp1]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(cfg);
    for p in poseidon2_table_provers_d5(Poseidon2Config::KOALA_BEAR_D1_W16) {
        prover.register_table_prover(p);
    }

    let mint_proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover
        .verify_all_tables::<EF5>(&mint_proof)
        .expect("mint-shaped proof verifies");
    let mint_shape = BatchStarkShape::<KoalaBear>::from_proof_with_fri(&mint_proof);
    assert_eq!(mint_shape.non_primitives.len(), 1);
    let mint_npo_rows = mint_shape.non_primitives[0].rows;
    assert!(mint_npo_rows < 16, "mint-shaped poseidon2 should be small");

    // First padded proof freezes the transfer-like FRI-locked profile.
    let mut transfer_like = mint_shape.clone();
    transfer_like.non_primitives[0].rows = 16;
    let mut traces_a = traces.clone();
    let mut data_a = CircuitProverData::new(
        ProverData::from_airs_and_degrees(
            &p3_circuit_prover::config::koala_bear(),
            &airs,
            &degrees,
        ),
        circuit_prover_data.primitive_columns.clone(),
        circuit_prover_data.non_primitive_columns.clone(),
    );
    data_a
        .pad_to_profile(&mut traces_a, &transfer_like)
        .unwrap();
    assert_eq!(
        traces_a
            .non_primitive_traces
            .get(&transfer_like.non_primitives[0].op_type)
            .unwrap()
            .rows(),
        16
    );

    let padded_a = prover.prove_all_tables(&traces_a, &data_a).unwrap();
    prover
        .verify_all_tables::<EF5>(&padded_a)
        .expect("padded mint→transfer-like proof verifies");
    let frozen = BatchStarkShape::<KoalaBear>::from_proof_with_fri(&padded_a);
    assert_eq!(frozen.non_primitives[0].rows, 16);
    assert_ne!(
        frozen.fri.as_ref().map(|f| f.commit_phase_len),
        mint_shape.fri.as_ref().map(|f| f.commit_phase_len),
        "padding NPO rows must change the FRI layout"
    );

    // Sol's sequencing: raw FRI-locked class union is structural (query
    // identity). Growable union is from_proof without FRI; only the padded
    // locked re-extraction is allowed to seed recurrence.
    assert!(
        matches!(
            mint_shape.union(&frozen),
            Err(ProofMetadataError::ProfileStructuralMismatch(_))
        ),
        "FRI-locked mint ∪ padded must reject query identity, not union"
    );
    let unlocked_mint = BatchStarkShape::<KoalaBear>::from_proof(&mint_proof);
    let unlocked_padded = BatchStarkShape::<KoalaBear>::from_proof(&padded_a);
    let unlocked_union = unlocked_mint.union(&unlocked_padded).unwrap();
    assert!(unlocked_union.fri.is_none());
    assert_eq!(unlocked_union.non_primitives[0].rows, 16);

    // Re-extracting a second independently padded mint proof is byte-identical
    // to the frozen FRI-locked profile.
    let mut traces_b = traces;
    let mut data_b = CircuitProverData::new(
        ProverData::from_airs_and_degrees(
            &p3_circuit_prover::config::koala_bear(),
            &airs,
            &degrees,
        ),
        circuit_prover_data.primitive_columns,
        circuit_prover_data.non_primitive_columns,
    );
    data_b.pad_to_profile(&mut traces_b, &frozen).unwrap();
    let padded_b = prover.prove_all_tables(&traces_b, &data_b).unwrap();
    prover
        .verify_all_tables::<EF5>(&padded_b)
        .expect("second padded proof verifies");
    let extracted = BatchStarkShape::<KoalaBear>::from_proof_with_fri(&padded_b);
    assert_eq!(
        extracted, frozen,
        "re-extraction of the padded proof must be byte-identical to the frozen FRI-locked profile"
    );
    assert_eq!(extracted.fri, frozen.fri);
}
