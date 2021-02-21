using System;
using System.Collections.Generic;

namespace Perceptomancer
{
    public class TraningSet
    {
        public List<double[]> InputList;
        public List<double[]> OutputList;

        public TraningSet(List<double[]> inputValues, List<double[]> outputValues)
        {
            InputList = inputValues;
            OutputList = outputValues;
        }

        public static TraningSet XorList()
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
    }
}
