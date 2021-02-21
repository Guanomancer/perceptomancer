using System;
using System.Collections.Generic;

namespace Perceptomancer
{
    [Serializable]
    public abstract class DeepLayer
    {
        protected static Random RandomNumberGenerator = new Random(1);

        public readonly IActivationFunction ActivationFunction;

        public readonly int NumberOfNeurons;
        public readonly List<DeepNeuron> Neurons = new List<DeepNeuron>();

        public double[] Output;
        public int NumberOfInputs;

        public DeepLayer(int numberOfNeurons, IActivationFunction activationFunction)
        {
            ActivationFunction = activationFunction;
            NumberOfNeurons = numberOfNeurons;
            Output = new double[NumberOfNeurons];
        }

        public abstract void Attach(DeepLayer previousLayer = null);

        public abstract double[] Feed(double[] inputValues);
    }
}