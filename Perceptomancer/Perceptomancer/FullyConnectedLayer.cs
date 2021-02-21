using System;

namespace Perceptomancer
{
    public class FullyConnectedLayer : DeepLayer
    {
        private static Random _rng = new Random(1);

        public FullyConnectedLayer(int numberOfNeurons)
            : base(numberOfNeurons)
        { }

        public override void Attach(DeepLayer previousLayer = null)
        {
            if (previousLayer == null)
                throw new ArgumentNullException("previousLayer");

            NumberOfInputs = previousLayer.NumberOfNeurons;
            for (int i = 0; i < NumberOfNeurons; i++)
                Neurons.Add(new Neuron(NumberOfInputs, _rng));
        }

        public override double[] Feed(double[] inputValues)
        {
            for(int i = 0; i < NumberOfNeurons; i++)
                Output[i] = Neurons[i].Activate(inputValues);

            return Output;
        }
    }
}
