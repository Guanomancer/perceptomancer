using System;

namespace Perceptomancer
{
    public class InputLayer : DeepLayer
    {
        private static Random _rng = new Random(1);

        public InputLayer(int numberOfNeurons)
            : base(numberOfNeurons)
        {
            NumberOfInputs = numberOfNeurons;
        }

        public override void Attach(DeepLayer previousLayer = null)
        {
            NumberOfInputs = previousLayer == null ?
                NumberOfInputs : previousLayer.NumberOfNeurons;
            for (int i = 0; i < NumberOfNeurons; i++)
                Neurons.Add(new Neuron(NumberOfInputs, _rng));
        }

        public override double[] Feed(double[] inputValues)
        {
            Output = inputValues;

            return Output;
        }
    }
}
