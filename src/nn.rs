use crate::engine::Value;

use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

pub trait Module {
    // Retrieve all trainable parameters as a vector.
    fn parameters(&self) -> Vec<&Value>;

    // Perform a forward pass through the module.
    fn forward(&self, inputs: &[Value]) -> Vec<Value>;
}


impl Neuron {

    /// Constructs a new `Neuron` with randomly initialized weights and a bias.
    ///
    /// # Parameters
    /// - `nin`: The number of input connections (features) to the neuron.
    ///
    /// # Returns
    /// Returns a `Neuron` instance with `nin` weights and one bias, all initialized to random values between -1.0 and 1.0.
    pub fn new(nin: usize) -> Neuron {
        let mut rng = thread_rng();
        let mut rand_value_fn = || {
            let data = rng.gen_range(-1.0..1.0);
            Value::from(data)
        };

        let mut w = Vec::new();
        for _ in 0..nin {
            w.push(rand_value_fn());
        }

        Neuron {
            w,
            b: rand_value_fn().add_label("b"),
        }
    }

    /// Performs a forward pass of the neuron using the given inputs.
    ///
    /// # Parameters
    /// - `xs`: A vector of `Value` representing the inputs to the neuron.
    ///
    /// # Returns
    /// The output of the neuron as a `Value`, applying the tanh activation function to the weighted sum of inputs plus the bias.
    
    pub fn forward(&self, xs: &Vec<Value>) -> Value {
        let products = std::iter::zip(&self.w, xs)
            .map(|(a, b)| a * b)
            .collect::<Vec<Value>>();

        let sum = self.b.clone() + products.into_iter().reduce(|acc, prd| acc + prd).unwrap();
        sum.tanh()
    }

    /// Retrieves all parameters (weights and bias) of the neuron.
    ///
    /// # Returns
    /// A vector of `Value` containing the neuron's bias followed by its weights.
    pub fn parameters(&self) -> Vec<Value> {
        [self.b.clone()]
            .into_iter()
            .chain(self.w.clone())
            .collect()
    }
}

impl MLP {
    /// Constructs a new `MLP` (Multi-Layer Perceptron) network.
    ///
    /// # Parameters
    /// - `nin`: The number of inputs to the network.
    /// - `nout`: A vector specifying the number of neurons in each layer of the network.
    ///
    /// # Returns
    /// Returns an `MLP` instance with layers defined by `nin` and `nout`.
    pub fn new(nin: usize, nout: Vec<usize>) -> MLP {
        let nout_len = nout.len();
        let layer_sizes: Vec<usize> = [nin].into_iter().chain(nout).collect();

        MLP {
            layers: (0..nout_len)
                .map(|i| Layer::new(layer_sizes[i], layer_sizes[i + 1]))
                .collect(),
        }
    }

    /// Performs a forward pass through the entire MLP network.
    ///
    /// # Parameters
    /// - `xs`: A vector of `Value` representing the input to the network.
    ///
    /// # Returns
    /// A vector of `Value` representing the output from the final layer of the network.
    pub fn forward(&self, mut xs: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            xs = layer.forward(&xs);
        }
        xs
    }

    /// Retrieves all trainable parameters from every layer of the MLP.
    ///
    /// # Returns
    /// A flattened vector of `Value` containing all the parameters of the network.
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl Layer {
    /// Constructs a new `Layer` consisting of multiple neurons.
    ///
    /// # Parameters
    /// - `nin`: The number of inputs each neuron in the layer should accept.
    /// - `nout`: The number of neurons in the layer.
    ///
    /// # Returns
    /// Returns a `Layer` instance containing `nout` neurons, each with `nin` inputs.
    pub fn new(nin: usize, nout: usize) -> Layer {
        Layer {
            neurons: (0..nout)
                .map(|_| Neuron::new(nin))
                .collect(),
        }
    }

    /// Performs a forward pass through the layer using the given inputs.
    ///
    /// # Parameters
    /// - `xs`: A vector of `Value` representing the inputs to the layer.
    ///
    /// # Returns
    /// A vector of `Value` representing the output from each neuron in the layer.
    pub fn forward(&self, xs: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(xs)).collect()
    }

    /// Retrieves all parameters (weights and biases) from all neurons in the layer.
    ///
    /// # Returns
    /// A flattened vector of `Value` containing all the parameters of the layer.
    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}
