extern crate ndarray as nd;

use nd::{prelude::*, Array};
use rand::{prelude::*, SeedableRng};

fn main() {
    let (n, m) = (4, 3);
    let learnrate = 0.1;
    let mut rng = thread_rng();

    let mut input = Array::<f32, _>::zeros((n, m));
    input = array![
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0f32]
    ];
    input.mapv_inplace(|x| sigmoid(x));
    //println!("input:\n{}",input);

    let mut output = Array::<f32, _>::ones((n, 1));
    output[[0, 0]] = 0f32;
    output[[3, 0]] = 0f32;
    //println!("output:\n{}",output);

    let mut weights = Array::<f32, Ix2>::zeros((m, 1));
    for x in 0..m {
        weights[[x, 0]] = rng.gen_range(0.0, 1.0);
    }
    //println!("weights:\n{}",weights);

    for _ in 0..10000 {
        let output_hat = input.dot(&weights).map(|x| sigmoid(*x));
        //println!("output_hat:\n{}",output_hat);

        let err_term =
            (&output - &output_hat) * &output_hat * (&output_hat.map(|x| 1.0 - x)) / learnrate;
        //println!("err_term:\n{}",err_term);

        let delta_w = input.t().dot(&err_term);
        //println!("delta_w:\n{}",delta_w);

        weights = weights + delta_w;
        //println!("weights:\n{}",weights);

        let error = (output.clone() - output_hat.clone())
            .mapv(|x| x.powf(2.0))
            .sum()
            / (2.0 * n as f32);
        //println!("error: {}",error);
    }

    let test_vec = array![[1.0, 1.0, 0.8]];
    let result = sigmoid(test_vec.dot(&weights)[[0, 0]]);
    println!("result: {}", result);
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
