use picograd::{Value, MLP};

fn main() {
    let network = MLP::new(4, vec![4, 4, 1]);

    let training_inputs = vec![
        vec![2.0, 3.0, 11.0],
        vec![30.0, 1.0, 0.5],
        vec![5.5, 1.0, 6.0],
        vec![11.0, 1.0, 1.0],
    ];

    let target_outputs = vec![1.0, 2.0, 3.0, 2.0];

    for _ in 0..100 {
        // Forward pass
        let predicted_values: Vec<Value> = training_inputs
            .iter()
            .map(|x| network.forward(x.iter().map(|x| Value::from(*x)).collect())[0].clone())
            .collect();
        let predictions: Vec<f64> = predicted_values.iter().map(|v| v.data()).collect();

        // Loss function
        let _yg = target_outputs.iter().map(|y| Value::from(*y));
        let loss: Value = predicted_values
            .into_iter()
            .zip(_yg)
            .map(|(yp, yg)| (yp - yg).pow(&Value::from(2.0)))
            .sum();

        println!("Loss: {} Predictions: {:?}", loss.data(), predictions);

        // Backward pass
        network.parameters().iter().for_each(|p| p.zero_grad());
        loss.backward();

        // Adjustment
        network.parameters().iter().for_each(|p| p.adjust(-0.05));
    }
}
