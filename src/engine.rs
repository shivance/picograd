use std::cell::{Ref, RefCell};
use std::iter::Sum;
use std::ops::{Add, Deref, Mul, Neg, Sub};
use std::rc::Rc;

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

type PropagateFn = fn(value: &Ref<_Value>);

pub struct _Value {
    data: f64,
    grad: f64,
    _op: Option<String>,
    _prev: Vec<Value>,
    propagate: Option<PropagateFn>,
    label: Option<String>,
}

impl _Value {
    /// Constructs a new `_Value` instance.
    /// 
    /// # Parameters
    /// - `data`: The numerical value of the `_Value`.
    /// - `label`: Optional label describing the `_Value`.
    /// - `op`: Optional string describing the operation that generated this `_Value`.
    /// - `prev`: Vector of previous `_Value` instances linked to this value.
    /// - `propagate`: Optional function for propagating gradients back through the network.
    ///
    /// # Returns
    /// Returns a new `_Value` with initialized fields.
    fn new(
        data: f64,
        label: Option<String>,
        op: Option<String>,
        prev: Vec<Value>,
        propagate: Option<PropagateFn>,
    ) -> _Value {
        _Value {
            data,
            grad: 0.0,
            label,
            _op: op,
            _prev: prev,
            propagate,
        }
    }
}

impl PartialEq for _Value {
    /// Checks equality between two `_Value` instances.
    /// 
    /// # Parameters
    /// - `other`: The other `_Value` instance to compare.
    ///
    /// # Returns
    /// Returns `true` if both instances have the same data, gradient, label, operation, and previous values.
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.grad == other.grad
            && self.label == other.label
            && self._op == other._op
            && self._prev == other._prev
    }
}

impl Eq for _Value {}

impl Hash for _Value {
    /// Provides a method to generate a hash for `_Value` instances.
    /// 
    /// # Parameters
    /// - `state`: The state of the hash function, which this method will modify.
    ///
    /// # Details
    /// This method hashes all significant fields of `_Value`, allowing it to be used in hashed collections.

    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.grad.to_bits().hash(state);
        self.label.hash(state);
        self._op.hash(state);
        self._prev.hash(state);
    }
}

impl Debug for _Value {
    /// Formats the `_Value` using the given formatter.
    /// 
    /// # Parameters
    /// - `f`: The formatter.
    ///
    /// # Returns
    /// Returns a `Result` that indicates whether the formatting was successful.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("_Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("label", &self.label)
            .field("_op", &self._op)
            .field("_prev", &self._prev)
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<_Value>>);

impl Value {
    /// Constructs a `Value` from a type that can be converted into `Value`.
    /// 
    /// # Type Parameters
    /// - `T`: The type that can be converted into `Value`.
    ///
    /// # Parameters
    /// - `t`: An instance of `T`.
    ///
    /// # Returns
    /// A new `Value` instance.
    pub fn from<T>(t: T) -> Value
    where
        T: Into<Value>,
    {
        t.into()
    }

    /// Creates a new `Value` containing a `_Value`.
    /// 
    /// # Parameters
    /// - `value`: The `_Value` to be encapsulated within `Value`.
    ///
    /// # Returns
    /// A new `Value` that wraps the given `_Value` in `Rc<RefCell<_Value>>`.

    fn new(value: _Value) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    /// Performs the backward propagation through the neural network graph.
    /// This function sets the initial gradient to 1.0 and recursively applies the
    /// propagation function defined in `_Value` nodes.
    pub fn backward(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.borrow_mut().grad = 1.0;
        fn _backward(visited: &mut HashSet<Value>, value: &Value) {
            if !visited.contains(&value) {
                visited.insert(value.clone());

                let borrowed_value = value.borrow();
                if let Some(propagate_fn) = borrowed_value.propagate {
                    propagate_fn(&borrowed_value);
                }

                for child_id in &value.borrow()._prev {
                    _backward(visited, child_id);
                }
            }
        }
        _backward(&mut visited, self);
    }

    /// Computes the power of `self` raised to `other` and returns a new `Value` representing the result.
    /// This also sets up the appropriate gradient propagation function.
    ///
    /// # Parameters
    /// - `other`: The exponent `Value`.
    ///
    /// # Returns
    /// A new `Value` representing the result of the power operation.
    
    pub fn pow(&self, other: &Value) -> Value {
        let result = self.borrow().data.powf(other.borrow().data);

        let propagate_fn: PropagateFn = |value| {
            let mut base = value._prev[0].borrow_mut();
            let power = value._prev[1].borrow();
            base.grad += power.data * (base.data.powf(power.data - 1.0)) * value.grad;
        };

        Value::new(_Value::new(
            result,
            None,
            Some("^".to_string()),
            vec![self.clone(), other.clone()],
            Some(propagate_fn),
        ))
    }

    /// Computes the hyperbolic tangent of `self` and returns a new `Value` representing the result.
    /// This also sets up the appropriate gradient propagation function.
    ///
    /// # Returns
    /// A new `Value` representing the result of the tanh operation.
    pub fn tanh(&self) -> Value {
        let result = self.borrow().data.tanh();

        let propagate_fn: PropagateFn = |value| {
            let mut _prev = value._prev[0].borrow_mut();
            _prev.grad += (1.0 - value.data.powf(2.0)) * value.grad;
        };

        Value::new(_Value::new(
            result,
            None,
            Some("tanh".to_string()),
            vec![self.clone()],
            Some(propagate_fn),
        ))
    }

    /// Adds a label to the `Value`.
    ///
    /// # Parameters
    /// - `label`: A string slice that is the label to add.
    ///
    /// # Returns
    /// The same `Value` with an updated label.
    pub fn add_label(self, label: &str) -> Value {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn zero_grad(&self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn adjust(&self, factor: f64) {
        let mut value = self.borrow_mut();
        value.data += factor * value.grad;
    }
}

impl Hash for Value {
    /// Implements the Hash trait for `Value` by delegating to the hash function of `_Value`.
    ///
    /// # Parameters
    /// - `state`: The mutable hasher state for writing the hash.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<_Value>>;

    /// Provides dereference access to the Rc<RefCell<_Value>>.
    ///
    /// # Returns
    /// A reference to the inner storage of the `Value`.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f64>> From<T> for Value {
    /// Converts a type that can be cast into a floating point number directly into a `Value`.
    ///
    /// # Type Parameters
    /// - `T`: The type which can be converted into `f64`.
    ///
    /// # Parameters
    /// - `t`: The value to convert.
    ///
    /// # Returns
    /// A new `Value` initialized with the converted floating-point value.

    fn from(t: T) -> Value {
        Value::new(_Value::new(t.into(), None, None, Vec::new(), None))
    }
}

impl Add<Value> for Value {
    type Output = Value;

    /// Adds two `Value` instances and returns a new `Value`.
    ///
    /// # Parameters
    /// - `other`: The other `Value` to add.
    ///
    /// # Returns
    /// A new `Value` representing the sum of self and other.
    fn add(self, other: Value) -> Self::Output {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    /// Adds two references of `Value` and returns a new `Value`.
    ///
    /// # Parameters
    /// - `other`: The reference to the other `Value` to add.
    ///
    /// # Returns
    /// A new `Value` representing the sum of self and other.
    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data + b.borrow().data;

    let propagate_fn: PropagateFn = |value| {
        let mut first = value._prev[0].borrow_mut();
        let mut second = value._prev[1].borrow_mut();

        first.grad += value.grad;
        second.grad += value.grad;
    };

    Value::new(_Value::new(
        result,
        None,
        Some("+".to_string()),
        vec![a.clone(), b.clone()],
        Some(propagate_fn),
    ))
}

impl Sub<Value> for Value {
    type Output = Value;

    /// Subtracts another `Value` from this one and returns a new `Value`.
    ///
    /// # Parameters
    /// - `other`: The other `Value` to subtract.
    ///
    /// # Returns
    /// A new `Value` representing the difference.
    fn sub(self, other: Value) -> Self::Output {
        add(&self, &(-other))
    }
}

impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;

    /// Subtracts a reference of another `Value` from a reference of this one and returns a new `Value`.
    ///
    /// # Parameters
    /// - `other`: The reference to the other `Value` to subtract.
    ///
    /// # Returns
    /// A new `Value` representing the difference.
    fn sub(self, other: &'b Value) -> Self::Output {
        add(self, &(-other))
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    /// Multiplies this `Value` by another and returns a new `Value`.
    ///
    /// # Parameters
    /// - `other`: The other `Value` to multiply.
    ///
    /// # Returns
    /// A new `Value` representing the product.
    fn mul(self, other: Value) -> Self::Output {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    /// Multiplies two references to `Value` instances and returns a new `Value`.
    ///
    /// # Parameters
    /// - `other`: A reference to another `Value` to multiply with this one.
    ///
    /// # Returns
    /// A new `Value` representing the product of `self` and `other`.
    fn mul(self, other: &'b Value) -> Self::Output {
        mul(self, other)
    }
}

fn mul(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data * b.borrow().data;

    let propagate_fn: PropagateFn = |value| {
        let mut first = value._prev[0].borrow_mut();
        let mut second = value._prev[1].borrow_mut();

        first.grad += second.data * value.grad;
        second.grad += first.data * value.grad;
    };

    Value::new(_Value::new(
        result,
        None,
        Some("*".to_string()),
        vec![a.clone(), b.clone()],
        Some(propagate_fn),
    ))
}

impl Neg for Value {
    type Output = Value;

    /// Negates this `Value` and returns a new `Value`.
    ///
    /// # Returns
    /// A new `Value` representing the negation of this `Value`.
    fn neg(self) -> Self::Output {
        mul(&self, &Value::from(-1))
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    /// Negates a reference to this `Value` and returns a new `Value`.
    ///
    /// # Returns
    /// A new `Value` representing the negation of this `Value`.
    fn neg(self) -> Self::Output {
        mul(self, &Value::from(-1))
    }
}

impl Sum for Value {
    /// Sums all elements in an iterator over `Value` and returns a single `Value` representing the sum.
    ///
    /// # Type Parameters
    /// - `I`: The iterator type, which must yield items of type `Value`.
    ///
    /// # Parameters
    /// - `iter`: The iterator over `Value` instances.
    ///
    /// # Returns
    /// A `Value` representing the sum of all items from the iterator.
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut sum = Value::from(0.0);
        loop {
            let val = iter.next();
            if val.is_none() {
                break;
            }

            sum = sum + val.unwrap();
        }
        sum
    }
}
