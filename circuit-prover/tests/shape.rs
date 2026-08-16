//! Value-free `BatchStarkShape` extraction, exact matching, pad-to-profile,
//! and profile-closure tests.

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
use p3_circuit_prover::shape::{
    BatchStarkShape, FriQueryShape, iterate_profile_closure, pad_traces_to_profile,
};
use p3_field::PrimeCharacteristicRing;

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
