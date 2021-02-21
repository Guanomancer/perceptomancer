using System;
using System.Collections.Generic;

namespace Perceptomancer
{
    public class TraningSet
    {
        public List<double[]> Input;
        public List<double[]> Output;

        public TraningSet(List<double[]> inputValues, List<double[]> outputValues)
        {
            Input = inputValues;
            Output = outputValues;
        }

        public static TraningSet Xor()
        {
            return new TraningSet(
                new List<double[]>
                {
                    new double[] { 0, 0 },
                    new double[] { 0, 1 },
                    new double[] { 1, 0 },
                    new double[] { 1, 1 },
                },
                new List<double[]>
                {
                    new double[] { 0 },
                    new double[] { 1 },
                    new double[] { 1 },
                    new double[] { 0 },
                }
                );
        }

        public static TraningSet XorInverse()
        {
            return new TraningSet(
                new List<double[]>
                {
                    new double[] { 0, 0 },
                    new double[] { 0, 1 },
                    new double[] { 1, 0 },
                    new double[] { 1, 1 },
                },
                new List<double[]>
                {
                    new double[] { 1 },
                    new double[] { 0 },
                    new double[] { 0 },
                    new double[] { 1 },
                }
                );
        }
    }
}
