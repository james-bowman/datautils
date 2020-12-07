package datautils_test

import (
	"math"
	"testing"

	"github.com/gonum/floats"
	"github.com/james-bowman/datautils"
)

var datasets = []struct {
	// ground truth relevance and predicted relevancy scores
	labels []float64
	probs  []float64
}{
	{
		labels: []float64{0, 0, 1, 1},
		probs:  []float64{0.1, 0.4, 0.35, 0.8},
	},
	{
		labels: []float64{0, 0, 1, 1, 0},
		probs:  []float64{0.1, 0.4, 0.35, 0.8, 0.85},
	},
	{
		labels: []float64{1, 0, 0, 1, 1, 0},
		probs:  []float64{0.02, 0.1, 0.4, 0.35, 0.8, 0.85},
	},
	{
		labels: []float64{0, 0},
		probs:  []float64{0.02, 0.1},
	},
	{
		labels: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		probs:  []float64{0.001485745854553862, 0.0014863790364460178, 0.0014863790364460178, 0.0014854873139097426, 0.001485745854553862, 0.001485745854553862, 0.0014863790364460178, 0.0014863790364460178, 0.001485745854553862, 0.001485745854553862, 0.0014863646408943988, 0.0014857314651254725, 0.0014857314651254725, 0.0014857314651254725, 0.0014863646408943988, 0.0014857314651254725, 0.0014863646408943988},
	},
}

func TestCumulativeGain(t *testing.T) {
	tests := []float64{2, 2, 3, 0, 0}

	for i, test := range tests {
		evaluation := datautils.NewRankingEvaluation(datasets[i].probs, datasets[i].labels)
		if test != evaluation.CumulativeGain(len(evaluation.Relevancies)) {
			t.Errorf("Test %d: Expected cumulative gain: %v but received %v", i+1, test, evaluation.CumulativeGain(len(evaluation.Relevancies)))
		}
	}
}

func TestDiscountedCumulativeGain(t *testing.T) {
	tests := []float64{1.5, 1.0616063116448504, 1.4178134987528725, 0, 0}

	for i, test := range tests {
		evaluation := datautils.NewRankingEvaluation(datasets[i].probs, datasets[i].labels)
		if test != evaluation.DiscountedCumulativeGain(len(evaluation.Relevancies), datautils.TraditionalRelevancy) {
			t.Errorf("Test %d: Expected discounted cumulative gain: %v but received %v", i+1, test, evaluation.DiscountedCumulativeGain(len(evaluation.Relevancies), datautils.TraditionalRelevancy))
		}
	}
}

func TestNormalisedDiscountedCumulativeGain(t *testing.T) {
	tests := []float64{0.9197207891481877, 0.6509209298071325, 0.6653497124326151, 1, 1}

	for i, test := range tests {
		evaluation := datautils.NewRankingEvaluation(datasets[i].probs, datasets[i].labels)
		if test != evaluation.NormalisedDiscountedCumulativeGain(len(evaluation.Relevancies), datautils.TraditionalRelevancy) {
			t.Errorf("Test %d: Expected normalised discounted cumulative gain: %v but received %v", i+1, test, evaluation.NormalisedDiscountedCumulativeGain(len(evaluation.Relevancies), datautils.TraditionalRelevancy))
		}
	}
}

func TestPrecisionRecallCurveCreation(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// expected
		precision  []float64
		recall     []float64
		thresholds []float64
	}{
		{
			precision:  []float64{2.0 / 3.0, 0.5, 1, 1},
			recall:     []float64{1, 0.5, 0.5, 0},
			thresholds: []float64{0.35, 0.4, 0.8},
		},
		{
			precision:  []float64{0.5, 1.0 / 3.0, 0.5, 0, 1},
			recall:     []float64{1, 0.5, 0.5, 0, 0},
			thresholds: []float64{0.35, 0.4, 0.8, 0.85},
		},
		{
			precision:  []float64{0.5, 0.4, 0.5, 1.0 / 3.0, 0.5, 0, 1},
			recall:     []float64{1, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0, 0},
			thresholds: []float64{0.02, 0.1, 0.35, 0.4, 0.8, 0.85},
		},
		{
			precision:  []float64{1},
			recall:     []float64{0},
			thresholds: []float64{},
		},
		{
			precision:  []float64{1},
			recall:     []float64{0},
			thresholds: []float64{},
		},
	}

	for i, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(datasets[i].probs, datasets[i].labels)
		if !floats.Equal(test.precision, curve.Precision) {
			t.Errorf("Expected precision: %v but received %v", test.precision, curve.Precision)
		}
		if !floats.Equal(test.recall, curve.Recall) {
			t.Errorf("Expected recall: %v but received %v", test.recall, curve.Recall)
		}
		if !floats.Equal(test.thresholds, curve.Thresholds) {
			t.Errorf("Expected thresholds: %v but received %v", test.thresholds, curve.Thresholds)
		}
	}
}

func TestAveragePrecision(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// expected
		ap float64
	}{
		{ap: 0.8333333333333333},
		{ap: 0.500000},
		{ap: 0.500000},
		{ap: 0.0},
		{ap: 0.0},
	}

	for i, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(datasets[i].probs, datasets[i].labels)
		ap := curve.AveragePrecision()
		if ap != test.ap {
			t.Errorf("Expected AP: %f but received %f", test.ap, ap)
		}
	}
}

func TestAverageInterpolatedPrecision(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// expected
		ap float64
	}{
		{ap: 0.848485},
		{ap: 0.545455},
		{ap: 0.545455},
		{ap: 0.090909},
		{ap: 0.090909},
	}

	for i, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(datasets[i].probs, datasets[i].labels)
		ap := curve.AverageInterpolatedPrecision()

		if math.Abs(ap-test.ap) > 0.000001 {
			t.Errorf("Expected AInterpP: %.6f but received %.6f", test.ap, ap)
		}
	}
}

func TestPrecisionAtK(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// expected
		precisions []float64
	}{
		{precisions: []float64{1, 1, 0.5, 2.0 / 3.0}},
		{precisions: []float64{1, 0, 0.5, 1.0 / 3.0, 0.5}},
		{precisions: []float64{1, 0, 0.5, 1.0 / 3.0, 0.5, 0.4, 0.5}},
		{precisions: []float64{1}},
		{precisions: []float64{1}},
	}

	for i, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(datasets[i].probs, datasets[i].labels)
		for k, v := range test.precisions {
			pAtk := curve.PrecisionAt(k)
			if pAtk != v {
				t.Errorf("Test %d. Expected P@%d: %f but received %f", i, k, v, pAtk)
			}
		}
	}
}

func TestRPrecision(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// expected
		rprecision float64
	}{
		{rprecision: 0.5},
		{rprecision: 0.5},
		{rprecision: 1.0 / 3.0},
		{rprecision: 1},
		{rprecision: 1},
	}

	for i, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(datasets[i].probs, datasets[i].labels)
		RP := curve.RPrecision()
		if RP != test.rprecision {
			t.Errorf("Expected RPrecision: %f but received %f", test.rprecision, RP)
		}
	}
}

func TestInterpolatedPrecisionAtR(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// expected
		precisions []float64
	}{
		{precisions: []float64{1, 1, 1, 1, 1, 1, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}},
		{precisions: []float64{1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
		{precisions: []float64{1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
		{precisions: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{precisions: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	}

	for i, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(datasets[i].probs, datasets[i].labels)
		for r, v := range test.precisions {
			pAtr := curve.InterpolatedPrecisionAt(float64(r) / 10)
			if pAtr != v {
				t.Errorf("Test %d. Expected Interp. P@%.1f: %f but received %f", i, float64(r)/10, v, pAtr)
			}
		}
	}
}
