package datautils_test

import (
	"testing"

	"github.com/gonum/floats"
	"github.com/james-bowman/datautils"
)

func TestPrecisionRecall(t *testing.T) {
	// Test the metric functions
	tests := []struct {
		// inputs
		labels []float64
		probs  []float64
		// expected
		precision  []float64
		recall     []float64
		thresholds []float64
		ap         float64
	}{
		{
			labels:     []float64{0, 0, 1, 1},
			probs:      []float64{0.1, 0.4, 0.35, 0.8},
			precision:  []float64{0.6666666666666666, 0.5, 1, 1},
			recall:     []float64{1, 0.5, 0.5, 0},
			thresholds: []float64{0.35, 0.4, 0.8},
			ap:         0.8333333333333333,
		},
		{
			labels:     []float64{0, 0, 1, 1, 0},
			probs:      []float64{0.1, 0.4, 0.35, 0.8, 0.85},
			precision:  []float64{0.5, 0.3333333333333333, 0.5, 0, 1},
			recall:     []float64{1, 0.5, 0.5, 0, 0},
			thresholds: []float64{0.35, 0.4, 0.8, 0.85},
			ap:         0.500000,
		},
		{
			labels:     []float64{1, 0, 0, 1, 1, 0},
			probs:      []float64{0.02, 0.1, 0.4, 0.35, 0.8, 0.85},
			precision:  []float64{0.5, 0.4, 0.5, 0.3333333333333333, 0.5, 0, 1},
			recall:     []float64{1, 0.6666666666666666, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333, 0, 0},
			thresholds: []float64{0.02, 0.1, 0.35, 0.4, 0.8, 0.85},
			ap:         0.500000,
		},
		{
			labels:     []float64{0, 0},
			probs:      []float64{0.02, 0.1},
			precision:  []float64{1},
			recall:     []float64{0},
			thresholds: []float64{},
			ap:         0.0,
		},
		{
			labels:     []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			probs:      []float64{0.001485745854553862, 0.0014863790364460178, 0.0014863790364460178, 0.0014854873139097426, 0.001485745854553862, 0.001485745854553862, 0.0014863790364460178, 0.0014863790364460178, 0.001485745854553862, 0.001485745854553862, 0.0014863646408943988, 0.0014857314651254725, 0.0014857314651254725, 0.0014857314651254725, 0.0014863646408943988, 0.0014857314651254725, 0.0014863646408943988},
			precision:  []float64{1},
			recall:     []float64{0},
			thresholds: []float64{},
			ap:         0.0,
		},
	}

	for _, test := range tests {
		curve := datautils.NewPrecisionRecallCurve(test.probs, test.labels)
		if !floats.Equal(test.precision, curve.Precision) {
			t.Errorf("Expected precision: %v but received %v", test.precision, curve.Precision)
		}
		if !floats.Equal(test.recall, curve.Recall) {
			t.Errorf("Expected recall: %v but received %v", test.recall, curve.Recall)
		}
		if !floats.Equal(test.thresholds, curve.Thresholds) {
			t.Errorf("Expected thresholds: %v but received %v", test.thresholds, curve.Thresholds)
		}
		ap := curve.AveragePrecision()
		if ap != test.ap {
			t.Errorf("Expected AP: %f but received %f", test.ap, ap)
		}
	}
}
