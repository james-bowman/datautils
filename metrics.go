package datautils

import (
	"fmt"
	"image/color"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

type PrecisionRecallCurve struct {
	Precision  []float64
	Recall     []float64
	Thresholds []float64
}

func NewPrecisionRecallCurve(predictions, labels []float64) PrecisionRecallCurve {
	if len(predictions) != len(labels) {
		panic("Prediction/Label length mismatch")
	}

	thresholds := make([]float64, len(predictions))
	recall := make([]float64, len(predictions))
	precision := make([]float64, len(predictions))
	ind := make([]int, len(predictions))

	// count total positives from ground truth
	pos := floats.Sum(labels)
	if pos == 0 {
		return PrecisionRecallCurve{
			Precision:  append(precision[:0], 1),
			Recall:     append(recall[:0], 0),
			Thresholds: thresholds[:0],
		}
	}

	copy(thresholds, predictions)
	floats.Argsort(thresholds, ind)

	var hits int
	var k int

	for i := len(thresholds) - 1; i >= 0; i-- {
		if labels[ind[i]] == 1 {
			hits++
		}
		recall[k] = float64(hits) / float64(pos)
		precision[k] = float64(hits) / float64(k+1)
		if recall[k] == 1 {
			break
		}
		k++
	}
	precision = precision[:k+1]
	recall = recall[:k+1]

	floats.Reverse(precision)
	floats.Reverse(recall)

	return PrecisionRecallCurve{
		Precision:  append(precision, 1),
		Recall:     append(recall, 0),
		Thresholds: thresholds[len(thresholds)-k-1:],
	}
}

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
