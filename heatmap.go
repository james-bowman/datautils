package datautils

import (
	"fmt"
	"image/color"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

type heatmap struct {
	x mat.Matrix
}

func (h heatmap) Dims() (c, r int)   { r, c = h.x.Dims(); return c, r }
func (h heatmap) Z(c, r int) float64 { return h.x.At(r, c) }
func (h heatmap) X(c int) float64    { return float64(c) }
func (h heatmap) Y(r int) float64    { return float64(r) }

type ticks []string

func (t ticks) Ticks(min, max float64) []plot.Tick {
	var retval []plot.Tick
	for i := math.Trunc(min); i <= max; i++ {
		retval = append(retval, plot.Tick{Value: i, Label: t[int(i)]})
	}
	return retval
}

func PlotHeatmap(corr mat.Matrix, xlabels []string, ylabels []string) (p *plot.Plot, err error) {
	pal := palette.Heat(48, 1)
	m := heatmap{corr}
	hm := plotter.NewHeatMap((plotter.GridXYZ)(m), pal)
	if p, err = plot.New(); err != nil {
		return
	}
	hm.NaN = color.RGBA{0, 0, 0, 0}

	p.Add(hm)
	p.X.Tick.Label.Rotation = 1.5
	p.Y.Tick.Label.Font.Size = 6
	p.X.Tick.Label.Font.Size = 6
	p.X.Tick.Label.XAlign = draw.XRight
	p.X.Tick.Marker = ticks(xlabels)
	p.Y.Tick.Marker = ticks(ylabels)

	l, err := plot.NewLegend()
	if err != nil {
		return p, err
	}

	thumbs := plotter.PaletteThumbnailers(pal)

	for i := len(thumbs) - 1; i >= 0; i-- {
		t := thumbs[i]
		if i != 0 && i != len(thumbs)-1 {
			l.Add("", t)
			continue
		}
		var val float64
		switch i {
		case 0:
			val = hm.Min
		case len(thumbs) - 1:
			val = hm.Max
		}
		l.Add(fmt.Sprintf("%.2g", val), t)
	}
	l.Left = true
	l.XOffs = -5
	l.ThumbnailWidth = 5
	l.Font.Size = 5

	p.Legend = l
	return
}
