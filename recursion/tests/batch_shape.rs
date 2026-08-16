//! Recursive `verify_p3_batch_shape_circuit` and packing-side exact-shape gate.

mod common;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::shape::BatchStarkShape;
use p3_circuit_prover::{BatchStarkProver, CircuitProverData, ConstraintProfile, TablePacking};
use p3_lookup::logup::LogUpGadget;
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::Poseidon2Config;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::verifier::{verify_p3_batch_proof_circuit, verify_p3_batch_shape_circuit};
use p3_test_utils::koala_bear_params::*;

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

fn tiny_batch_proof() -> (
    p3_circuit_prover::batch_stark_prover::BatchStarkProof<MyConfig>,
    CircuitProverData<MyConfig>,
    MyConfig,
) {
    let mut builder = CircuitBuilder::new();
    let expected = builder.alloc_public_input("sum");
    let a = builder.alloc_const(F::from_u64(1), "one");
    let b = builder.alloc_const(F::from_u64(2), "two");
    let sum = builder.add(a, b);
    builder.connect(sum, expected);
    let circuit = builder.build().unwrap();

    let table_packing = TablePacking::new(2, 4);
    let config = make_test_config();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[F::from_u64(3)]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(make_test_config()).with_table_packing(table_packing);
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables::<F>(&proof).unwrap();
    (proof, circuit_prover_data, config)
}

#[test]
fn verify_p3_batch_shape_circuit_matches_and_packs() {
    let (batch_stark_proof, circuit_prover_data, config) = tiny_batch_proof();
    let common = circuit_prover_data.common_data();
    let shape = BatchStarkShape::<F>::from_proof(&batch_stark_proof);

    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let lookup_gadget = LogUpGadget::new();
    const TRACE_D: usize = 1;

    let mut circuit_builder = CircuitBuilder::new();
    let poseidon2_perm = default_koalabear_poseidon2_16();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        poseidon2_perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    let (verifier_inputs, _mmcs_op_ids) = verify_p3_batch_shape_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D,
    >(
        &config,
        &mut circuit_builder,
        &shape,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &[],
    )
    .unwrap();

    let verification_circuit = circuit_builder.build().unwrap();
    let num_tables = common
        .preprocessed
        .as_ref()
        .map(|g| g.instances.len())
        .unwrap_or(0);
    let pis: Vec<Vec<F>> = vec![vec![]; num_tables];
    let (public_inputs, _private_inputs) = verifier_inputs
        .pack_values_matching_shape(&pis, &batch_stark_proof, common, &shape)
        .unwrap();
    assert_eq!(public_inputs.len(), verification_circuit.public_flat_len);

    let mut bad = shape.clone();
    let mut rows = bad.rows.as_array();
    rows[0] += 2;
    bad.rows = p3_circuit_prover::batch_stark_prover::RowCounts::new(rows);
    assert!(
        verifier_inputs
            .pack_values_matching_shape(&pis, &batch_stark_proof, common, &bad)
            .is_err()
    );
}

#[test]
fn verify_p3_batch_shape_circuit_rejects_mismatched_shape() {
    let (batch_stark_proof, circuit_prover_data, config) = tiny_batch_proof();
    let common = circuit_prover_data.common_data();
    let mut shape = BatchStarkShape::<F>::from_proof(&batch_stark_proof);
    shape.alu_variant = p3_circuit_prover::batch_stark_prover::AirVariant::Baseline;

    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let lookup_gadget = LogUpGadget::new();
    let mut circuit_builder = CircuitBuilder::new();
    let err = match verify_p3_batch_shape_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        1,
    >(
        &config,
        &mut circuit_builder,
        &shape,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &[],
    ) {
        Ok(_) => panic!("mismatched shape should have been rejected"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("alu_variant") || msg.contains("Invalid proof shape"),
        "unexpected error: {msg}"
    );
}

#[test]
fn verify_p3_batch_proof_circuit_still_delegates() {
    let (batch_stark_proof, circuit_prover_data, config) = tiny_batch_proof();
    let common = circuit_prover_data.common_data();
    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        scalars.num_queries,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let lookup_gadget = LogUpGadget::new();
    let mut circuit_builder = CircuitBuilder::new();
    let poseidon2_perm = default_koalabear_poseidon2_16();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        poseidon2_perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        1,
    >(
        &config,
        &mut circuit_builder,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &[],
    )
    .unwrap();
}
