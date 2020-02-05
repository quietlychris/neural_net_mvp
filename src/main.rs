use ndarray::prelude::*;
use ndarray_linalg::types::Scalar;
use ndarray_linalg::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use std::f32;

//    let mut twos = Array::from_elem((4, 1), 1f32).map(|x| *x + 1.0);
//    let mut es = Array::from_elem((4, 1), 1f32).map(|x| *&x.exp());
//    println!("twos: {}",twos);
//    println!("e's: {}",es);

fn main() {
    let training_set_inputs = array![
        [0.0f32, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ];
    let training_set_outputs = array![[0.0, 1.0, 1.0, 0.0]].reversed_axes(); // .reversed_axes() to transponse
    //println!("training_set_outputs:\n{}", training_set_outputs);
    let mut synaptic_weights =
        Array2::<f32>::random((1, 3), Uniform::new(0., 1.)).map(|x| (2.0 * x) - 1.0);
    //println!("synaptic_weights:\n{}\n", synaptic_weights);
    let mut ones = Array::from_elem((4, 1), 1f32);

    let mut output = ones.clone()
        / ((ones.clone() + (training_set_inputs.dot(&synaptic_weights.t())).map(|x| *x * -1.0))
            .map(|x| *&x.exp()));
    //println!("output_zero:\n{}\n", output);

    for iteration in 0..10000 {
        output = ones.clone()
            / ((ones.clone()
                + (training_set_inputs.dot(&synaptic_weights.t())).map(|x| *x * -1.0))
            .map(|x| *&x.exp()));
        //println!("output_{}:\n{}\n", iteration, output);

        let dp_feeder = (&training_set_outputs - &output) * &output * (&output.map(|x| 1.0 - x));
        //println!("dp_feeder:\n{}\n", dp_feeder);
        let new_weights = (training_set_inputs.clone().reversed_axes()).dot(&dp_feeder);

        *&mut synaptic_weights = &synaptic_weights + &new_weights.reversed_axes();
        //println!("synaptic_weights #{}:\n{}\n", iteration, synaptic_weights);
    }

    let mut result_ones = Array::from_elem((3, 1), 1f32);
    let test_vec = array![[1.0, 1.0, 0.80]]; // Checking this array to see if it fits our training set
    let result = result_ones.clone()
        / (result_ones
            + (test_vec
                .dot(&synaptic_weights.reversed_axes())
                .map(|x| *x * -1.0))
            .map(|x| *&x.exp()));

    print!("result for {}:\n{}\n", &test_vec, result[[0, 0]]);
}
