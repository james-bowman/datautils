package datautils

import (
	"fmt"
	"image/color"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

// PrecisionRecallCurve represents a precision recall curve for visualising and measuring the performance of a
// classification or information retrieval model.  It can be used to evaluate how well the model predictions
// can be ranked compared to a perfect ranking according to the ground truth labels.  This is usefull when
// evaluating ranking based on relevancy for information retrieval or raw classification performance based on
// predicted probability of class membership e.g. logistic regression predictions without using a threshold to
// determine the class for the predicted probability.  To measure Precision@K using the precision recall curve
// where K is the cut-off simply index Precision e.g. Precision[k-1].
type PrecisionRecallCurve struct {
	// Precision is a slice containing the ranked precision values at K for the predictions until all positive/
	// relevant items were found according to corresponding the ground truth labels (recall==1)
	Precision []float64

	// Recall is a slice containing the ranked recall values at K for the predictions until all positive/
	// relevant items were found according to corresponding the ground truth labels (recall==1)
	Recall []float64

	// Thresholds is a slice containing the ranked (sorted) predictions (probability/similarity scores) until
	// all positive/relevant items were found according to corresponding the ground truth labels (recall==1)
	Thresholds []float64
}

// NewPrecisionRecallCurve creates a new precision recall curve.  The precision recall curve visualises how well
// the model's predictions (or similarity scores for information retrieval) can be ranked compared to a perfect
// ranking according to the ground truth labels.  Both the supplied predictions and labels slices can be in any
// order providing they are identical lengths and their order matches e.g. predictions[5] corresponds to the
// ground truth labels[5].  As Precision Recall curves and average precision (summarising the curve as a single
// metric/area under the curve) represent a binary class/relevance measure we assume that any label value greater
// than 0 represents a positive/relative observation (and 0 label values represent a negative/non-relevant
// observation).
func NewPrecisionRecallCurve(predictions, labels []float64) PrecisionRecallCurve {
	if len(predictions) != len(labels) {
		panic("Prediction/Label length mismatch")
	}

	thresholds := make([]float64, len(predictions))
	recall := make([]float64, len(predictions))
	precision := make([]float64, len(predictions))
	ind := make([]int, len(predictions))

	// count total positive/relevant observations from ground truth
	// as floats.Norm() does not work for zero norm we will use floats.Count() instead
	positives := floats.Count(func(x float64) bool {
		if x > 0 {
			return true
		}
		return false
	}, labels)

	if positives == 0 {
		return PrecisionRecallCurve{
			Precision:  append(precision[:0], 1),
			Recall:     append(recall[:0], 0),
			Thresholds: thresholds[:0],
		}
	}

	// rank predictions/similarities
	copy(thresholds, predictions)
	floats.Argsort(thresholds, ind)

	var hits int
	var k int

	for i := len(thresholds) - 1; i >= 0; i-- {
		// assume that any label value over 0 is positive/relevant.  Average precision works on a binary label
		// but in some cases we may use non-binary/multi-class labels e.g. for degrees of relevancy in information
		// retrieval
		if labels[ind[i]] > 0 {
			hits++
		}
		recall[k] = float64(hits) / float64(positives)
		precision[k] = float64(hits) / float64(k+1)
		if recall[k] == 1 {
			break
		}
		k++
	}
	// truncate precision and recall to where the last relevant/positive item was ranked (recall==1)
	precision = precision[:k+1]
	recall = recall[:k+1]

	// reverse order so highest similarity/probability is ranked higher/first
	floats.Reverse(precision)
	floats.Reverse(recall)

	// TODO: Discounted Culmulative Gain and Normalised Discounted Culmulative Gain metrics

	return PrecisionRecallCurve{
		Precision:  append(precision, 1),
		Recall:     append(recall, 0),
		Thresholds: thresholds[len(thresholds)-k-1:],
	}
}

// Plot renders the entire precision recall curve as a plot for visualisation.
func (c PrecisionRecallCurve) Plot() *plot.Plot {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	ap := c.AveragePrecision()

	p.Title.Text = fmt.Sprintf("Precision-recall Curve, AP=%f", ap)
	p.X.Label.Text = "Recall"
	p.Y.Label.Text = "Precision"

	pts := make(plotter.XYs, len(c.Precision))
	for i := range pts {
		pts[i].X = c.Recall[i]
		pts[i].Y = c.Precision[i]
	}

	line, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}
	line.Color = color.RGBA{R: 255, B: 128, A: 255}
	p.Add(line)

	return p
}

// AveragePrecision calculates the average precision based on the predictions and labels the curve was
// constructed with.  Average Precision represents the area under the curve of the precision recall curve
// and is a method for summarising the curve in a single metric.
func (c PrecisionRecallCurve) AveragePrecision() float64 {
	var sum float64
	for i := 0; i < len(c.Precision)-1; i++ {
		sum += (c.Recall[i+1] - c.Recall[i]) * c.Precision[i]
	}
	return -sum
}

type ConfusionMatrix struct {
	Observations, Pos, Neg, TruePos, TrueNeg, FalsePos, FalseNeg int
}

func NewConfusionMatrix(predictions []float64, labels []float64, threshold float64) ConfusionMatrix {
	var matrix ConfusionMatrix
	var y float64
	for i, v := range labels {
		matrix.Observations++

		if predictions[i] >= threshold {
			y = 1.0
		} else {
			y = 0.0
		}

		// evaluate result and collect stats
		if v == 1 {
			matrix.Pos++
			if y == 1 {
				matrix.TruePos++
			} else {
				matrix.FalseNeg++
			}
		} else {
			matrix.Neg++
			if y == 1 {
				matrix.FalsePos++
			} else {
				matrix.TrueNeg++
			}
		}
	}
	return matrix
}

func (c ConfusionMatrix) String() string {
	var s string

	horiz := "------------------------------------------------------------------------------------------------------\n"

	s = fmt.Sprintf("Observations = %-10d |       Predicted No       |       Predicted Yes      |\n", c.Observations)
	s = s + horiz
	s = fmt.Sprintf("%sActual No                 |       TN = %-10d    |       FP = %-10d    |\n", s, c.TrueNeg, c.FalsePos)
	s = fmt.Sprintf("%sActual Yes                |       FN = %-10d    |       TP = %-10d    |  Recall = %f\n", s, c.FalseNeg, c.TruePos, c.Recall())
	s = s + horiz
	s = fmt.Sprintf("%s                                                     |   Precision = %-10f |  Accuracy = %f\n", s, c.Precision(), c.Accuracy())
	s = fmt.Sprintf("%sF1 Score = %f\n", s, c.F1())

	return s
}

func (c ConfusionMatrix) Precision() float64 {
	return float64(c.TruePos) / float64(c.TruePos+c.FalsePos)
}

func (c ConfusionMatrix) Recall() float64 {
	return float64(c.TruePos) / float64(c.TruePos+c.FalseNeg)
}

func (c ConfusionMatrix) Accuracy() float64 {
	return float64(c.TrueNeg+c.TruePos) / float64(c.Observations)
}

func (c ConfusionMatrix) F1() float64 {
	return 2 * ((c.Precision() * c.Recall()) / (c.Precision() + c.Recall()))
}
