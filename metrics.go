package datautils

import (
	"fmt"
	"image/color"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

type DiscountedCumulativeGain struct {
	// Thresholds is a slice containing the ranked (sorted) predictions (probability/similarity scores) until
	// all positive/relevant items were found according to corresponding the ground truth labels (recall==1)
	Thresholds []float64
}

/*
func NewDiscountedCumulativeGain(predictions, labels []float64) DiscountedCumulativeGain {
	if len(predictions) != len(labels) {
		panic("Prediction/Label length mismatch")
	}

	thresholds := make([]float64, len(predictions))
	ind := make([]int, len(predictions))

	cg := make([]float64, len(predictions))
	dcg := make([]float64, len(predictions))
	perfectdcg := make([]float64, len(predictions))

	recall := make([]float64, len(predictions))
	precision := make([]float64, len(predictions))


	maxRelevance := floats.Max(labels)
	if maxRelevance == 0 {
		return DiscountedCumulativeGain{
			Thresholds: thresholds[:0],
		}
	}

	// rank predictions/similarities
	copy(thresholds, predictions)
	floats.Argsort(thresholds, ind)

	dcg[0] = labels[ind[len(ind)-1]]
	k := 1

	for i := len(ind) - 2; i >= 0; i-- {
		dcg[k] = dcg[k-1] + labels[ind[i]]/math.Log2(k)
		k++
	}

	for i := maxRelevance; i >= 0; i-- {
		count := floats.Count(func(x float64) bool {
			if x == i {
				return true
			}
			return false
		}, labels)

	}

	perfectdcg[0] = maxRelevance

	for i := 1; i < len(perfectdcg); i++ {

	}



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
*/

// PrecisionRecallCurve represents a precision recall curve for visualising and measuring the performance of a
// classification or information retrieval model.  It can be used to evaluate how well the model predictions
// can be ranked compared to a perfect ranking according to the ground truth labels.  This is usefull when
// evaluating ranking based on relevancy for information retrieval or raw classification performance based on
// predicted probability of class membership e.g. logistic regression predictions without using a threshold to
// determine the class for the predicted probability.
// It is important to note that Precision[0] and Recall[0] indicate the precision and recall @ 0 and so will always
// be 1 and 0 respectively.
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

	positives int
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
		return (x > 0)
	}, labels)

	if positives == 0 {
		return PrecisionRecallCurve{
			Precision:  append(precision[:0], 1),
			Recall:     append(recall[:0], 0),
			Thresholds: thresholds[:0],
			positives:  positives,
		}
	}

	// rank predictions/similarities
	copy(thresholds, predictions)
	floats.Argsort(thresholds, ind)

	var hits int
	var k int

	for i := len(ind) - 1; i >= 0; i-- {
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

	return PrecisionRecallCurve{
		Precision:  append(precision, 1),
		Recall:     append(recall, 0),
		Thresholds: thresholds[len(thresholds)-k-1:],
		positives:  positives,
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
	//	var sum float64

	var sum float64
	for i := 0; i < len(c.Precision)-1; i++ {
		sum += (c.Recall[i+1] - c.Recall[i]) * c.Precision[i]
	}
	return -sum
}

// AverageInterpolatedPrecision calculates the average interpolated precision based on the predictions and labels
// the curve was constructed with.  Average Interpolated Precision represents the area under the curve of the precision
// recall curve using interpolated precision for 11 fixed recall values {0.0, 0.1, 0.2, ... 1.0}.
func (c PrecisionRecallCurve) AverageInterpolatedPrecision() float64 {
	var sum float64
	for i := 0; i <= 10; i++ {
		sum += c.InterpolatedPrecisionAt(float64(i) / 10.0)
	}
	return sum / 11.0
}

// RPrecision returns the R-Precision.  The total number of relevant documents, R, is used as the cutoff
// for calculation, and this varies from query to query. It counts the number of results ranked above the
// cutoff that are relevant, r, and turns that into a relevancy fraction: r/R.
func (c PrecisionRecallCurve) RPrecision() float64 {
	return c.Precision[len(c.Precision)-1-c.positives]
}

// PrecisionAt calculates the Precision@k.  This represents the precision at a certain cut-off, k i.e.
// if a search returns 10 (k=10) results what is the proportion of those 10 results that are relevant or
// if we are only interested in the relevancy of the top ranked item (k=1) is that item relevant or not.
func (c PrecisionRecallCurve) PrecisionAt(k int) float64 {
	return c.Precision[len(c.Precision)-1-k]
}

// InterpolatedPrecisionAt calculates an interpolated Precision@r.  This can be used to calculate the precision for
// a specific recall value that does not necessarily occur explicitly in the ranking.  It is calculated by taking the
// maximum precision value over all recalls greater than r.
func (c PrecisionRecallCurve) InterpolatedPrecisionAt(r float64) float64 {
	// max precision [ recall > r]
	var inds []int
	var err error
	if inds, err = floats.Find(inds, func(x float64) bool { return (x >= r) }, c.Recall, -1); err != nil {
		panic("Failed to find items with recall higher " + err.Error())
	}

	var max float64
	for _, v := range inds {
		if c.Precision[v] > max {
			max = c.Precision[v]
		}
	}
	return max
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
