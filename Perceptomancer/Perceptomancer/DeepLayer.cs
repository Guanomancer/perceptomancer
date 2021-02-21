using System;
using System.Collections.Generic;

namespace Perceptomancer
{
    [Serializable]
    public abstract class DeepLayer
    {
        public readonly int NumberOfNeurons;
        public readonly List<Neuron> Neurons = new List<Neuron>();

        public double[] Output;
        public int NumberOfInputs;

        public DeepLayer(int numberOfNeurons)
        {
            NumberOfNeurons = numberOfNeurons;
            Output = new double[NumberOfNeurons];
        }

        public abstract void Attach(DeepLayer previousLayer = null);

        public abstract double[] Feed(double[] inputValues);
    }
}