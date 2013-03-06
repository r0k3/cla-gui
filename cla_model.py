import numpy as np
from scipy.interpolate import interp1d

from traits.api import (HasTraits, Int, Float, Array, Instance, Str, Property,
                        on_trait_change)
from chaco.api import (ArrayPlotData, Plot, LabelAxis, create_line_plot, create_scatter_plot,
                       HPlotContainer, GridDataSource, GridMapper, DataRange1D, BasePlotContainer,
                       DataRange2D, ColorBar, CMapImagePlot, LinearMapper, PlotAxis,
                       ImageData, RdBu as cmap)
from chaco.ticks import ShowAllTickGenerator
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from chaco.tools.api import PanTool
import enaml
from enaml.stdlib.sessions import show_simple_view
with enaml.imports():
    from cla_view import MainView

from CLA import CLA


class CLAModel(HasTraits):

    csvfile = Str
    current_sigma = Float
    sigma_percent = Int
    min_sigma = Float(0.0)
    max_sigma = Float(1.0)
    plot = Instance(Plot)
    bar = Instance(Plot)
    corr_plot = Instance(BasePlotContainer)
    cursor = Instance(BaseCursorTool)
    current_weight = Array
    weight_interpolater = Property
    sigma_index = Int(0)

    def __init__(self, *a, **kw):
        super(CLAModel, self).__init__(*a, **kw)
        self.load_data()
        self.solve_cla()

    def _plot_default(self):
        plot_data = ArrayPlotData(index=self.sigma, value=self.mu)
        self.plot_data = plot_data
        plot = Plot(data=plot_data)
        line = create_line_plot([self.sigma, self.mu], add_grid=True,
                                value_bounds=(min(self.mean), max(self.mean)),
                                add_axis=True, index_sort='ascending',
                                orientation='h')
        scatter = create_scatter_plot([np.sqrt(np.diag(self.covar)), np.squeeze(self.mean)],
                                      index_bounds=(line.index_range.low,
                                                    line.index_range.high),
                                      value_bounds=(line.value_range.low,
                                                    line.value_range.high),
                                      marker='circle', color='blue')
        plot.add(line)
        left, bottom = line.underlays[-2:]
        left.title = 'Return'
        bottom.title = 'Risk'
        plot.add(scatter)
        cursor = CursorTool(line, drag_button='left', color='blue')
        self.cursor = cursor
        #cursor.current_position = self.sigma[0], self.mu[0]
        line.overlays.append(cursor)
        line.tools.append(PanTool(line, drag_button='right'))
        #line.overlays.append(ZoomTool(line))
        return plot

    @on_trait_change('cursor:current_index')
    def update_cursor_pos(self):
        self.current_sigma = self.sigma[self.cursor.current_index]
        self.sigma_index = self.cursor.current_index

    def _sigma_index_changed(self):
        self.current_sigma = self.sigma[self.sigma_index]
        self.cursor.current_index = self.sigma_index

    def _bar_default(self):
        index = np.arange(0, len(self.current_weight))
        bar_data = ArrayPlotData(index=index, value=self.current_weight)
        self.bar_data = bar_data
        bar = Plot(data=bar_data)
        bar.plot(('index', 'value'), type='bar', bar_width=0.8, color='auto')
        label_axis = LabelAxis(bar, orientation='bottom', title='components',
                               tick_interval=1,
                               positions=index, labels=self.header, small_haxis_style=True)
        bar.underlays.remove(bar.index_axis)
        bar.index_axis = label_axis
        bar.range2d.y_range.high = 1.0
        return bar

    def _corr_plot_default(self):
        diag = self.covar.diagonal()
        corr = self.covar / np.sqrt(np.outer(diag, diag))
        N = len(diag)
        value_range = DataRange1D(low=-1, high=1)
        color_mapper = cmap(range=value_range)
        index = GridDataSource()
        value = ImageData()
        mapper = GridMapper(range=DataRange2D(index),
                            y_low_pos=1.0, y_high_pos=0.0)
        index.set_data(xdata=np.arange(-0.5, N),
                       ydata=np.arange(-0.5, N))
        value.set_data(np.flipud(corr))
        self.corr_data = value
        cmap_plot = CMapImagePlot(
            index=index,
            index_mapper=mapper,
            value=value,
            value_mapper=color_mapper,
            padding=(40, 40, 100, 40)
        )

        yaxis = PlotAxis(cmap_plot, orientation='left',
                         tick_interval=1,
                         tick_label_formatter=lambda x: self.header[int(N - 1 - x)],
                         tick_generator=ShowAllTickGenerator(positions=np.arange(N)))
        xaxis = PlotAxis(cmap_plot, orientation='top',
                         tick_interval=1,
                         tick_label_formatter=lambda x: self.header[int(x)],
                         tick_label_alignment='edge',
                         tick_generator=ShowAllTickGenerator(positions=np.arange(N)))
        cmap_plot.overlays.append(yaxis)
        cmap_plot.overlays.append(xaxis)
        colorbar = ColorBar(index_mapper=LinearMapper(range=cmap_plot.value_range),
                            plot=cmap_plot,
                            orientation='v',
                            resizable='v',
                            width=10,
                            padding=(40, 5, 100, 40))
        container = HPlotContainer(bgcolor='transparent')
        container.add(cmap_plot)
        container.add(colorbar)
        return container

    def load_data(self):
        with open(self.csvfile, 'rb') as f:
            self.header = f.readline().strip().split(',')
        data = np.genfromtxt(self.csvfile, delimiter=',', skip_header=1)
        self.mean = np.array(data[:1]).T
        self.lB = np.array(data[1:2]).T
        self.uB = np.array(data[2:3]).T
        self.covar = np.array(data[3:])

    def solve_cla(self):
        cla = CLA.CLA(self.mean, self.covar, self.lB, self.uB)
        cla.solve()
        self.cla = cla
        mu, sigma, weights = self.cla.efFrontier(100)
        # reverse order so sigma is high to low
        self.mu = np.array(mu[::-1])
        self.sigma = np.array(sigma[::-1])
        self.min_sigma = self.sigma.min()
        self.max_sigma = self.sigma.max()
        if self.current_sigma < self.min_sigma:
            self.current_sigma = self.min_sigma
        self.weights = np.hstack([np.array(w) for w in weights[::-1]])
        self.current_weight = self.weights[:,0]

    def sharpe_ratio(self):
        # TODO: use CLA.evalSR() to return Sharpe ratio
        pass

    def _get_weight_interpolater(self):
        return interp1d(self.sigma, self.weights)

    def set_current_weight(self):
        """ Interpolate the weights on the efficient frontier
        """
        try:
            self.current_weight = self.weight_interpolater(self.current_sigma)
            self.bar_data.set_data('value', self.current_weight)
        except Exception, e:
            pass

    def _current_sigma_changed(self):
        self.set_current_weight()

if __name__ == '__main__':
    filename = 'CLA/CLA_Data.csv'
    portfolio = CLAModel(csvfile=filename)
    view = MainView(model=portfolio)
    show_simple_view(view)
