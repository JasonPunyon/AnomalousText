using System;
using System.Linq;

namespace AnomalousText
{
    public class AnomalyScorer
    {
        private static int[] _topNTermBuckets = { 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, Int32.MaxValue };

        /// <summary>
        /// Calculates an anomaly score for each provided document using the "Distance To Textual Complement" method.<br />
        /// See http://nlp.shef.ac.uk/Completed_PhD_Projects/guthrie.pdf for the original research. <br />
        /// See http://jasonpunyon.com/blog/2014/09/02/a-wild-anomaly-appears/ for a laypersons explanation.
        /// </summary>
        /// <param name="documents">An array of arrays representing tokenized documents.</param>
        /// <returns>An anomaly score for each document in the order they were provided.</returns>
        public static double[] ScoreDocuments(string[][] documents)
        {
            var bucketNumbers = Enumerable.Range(0, _topNTermBuckets.Length).ToArray();
            var tokensToTopNBucket = documents
                .SelectMany(o => o)
                .GroupBy(o => o)
                .OrderByDescending(o => o.Count())
                .Select((o, i) => new { Token = o.Key, Bucket = bucketNumbers.SkipWhile(bucketNumber => i >= _topNTermBuckets[bucketNumber]).First() })
                .ToDictionary(o => o.Token, o => o.Bucket);

            var universeBucketCounts = documents
                .SelectMany(o => o)
                .GroupBy(o => tokensToTopNBucket[o])
                .ToDictionary(o => o.Key, o => o.Count());

            var universeVector = _topNTermBuckets.Select((o, i) => universeBucketCounts.ContainsKey(i) ? universeBucketCounts[i] : 0.0).ToArray();
            var universeTokenCount = universeVector.Sum();

            var documentVectors = documents
                .Select(o => o.GroupBy(p => tokensToTopNBucket[p]).ToDictionary(p => p.Key, p => p.Count()))
                .Select(o => _topNTermBuckets.Select((p, i) => o.ContainsKey(i) ? o[i] : 0.0).ToArray());

            return documentVectors
                .Select(document =>
                {
                    var documentTokenCount = document.Sum();
                    var complementTokenCount = universeTokenCount - documentTokenCount;
                    var complementVector = Subtract(universeVector, document).Select(o => o / complementTokenCount).ToArray();
                    var documentVector = document.Select(o => o / documentTokenCount).ToArray();
                    return ManhattanDistance(documentVector, complementVector);
                }).ToArray();
        }

        private static double[] Subtract(double[] left, double[] right)
        {
            return left.Zip(right, (l, r) => l - r).ToArray();
        }

        private static double ManhattanDistance(double[] left, double[] right)
        {
            return left.Zip(right, (l, r) => Math.Abs(l - r)).Sum();
        }
    }
}