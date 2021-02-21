using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptomancer.Tests
{
    [TestFixture]
    public class DeepNet_Tests
    {
        [Test]
        public void Ctor_DoesNotFail()
        {
            var net = new DeepNet();
        }
    }
}
