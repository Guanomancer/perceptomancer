using System;

namespace Perceptomancer
{
    public class InputLayer : DeepLayer
    {
        public InputLayer(int numberOfNeurons)
            : base(numberOfNeurons, null)
        {
            NumberOfInputs = numberOfNeurons;
        }

        public override void Attach(DeepLayer previousLayer = null)
        {
            NumberOfInputs = previousLayer == null ?
                NumberOfInputs : previousLayer.NumberOfNeurons;
            for (int i = 0; i < NumberOfNeurons; i++)
                Neurons.Add(new DeepNeuron(NumberOfInputs, ActivationFunction, RandomNumberGenerator));
        }

        public override double[] Feed(double[] inputValues)
        {
            Output = inputValues;

            return Output;
        }
    }
}
