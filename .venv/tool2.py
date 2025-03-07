import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import scipy.stats
from pubsub import pub
import wx
import yfinance as yf


class InputPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.stock_inputs = []
        self.weight_inputs = []

        main_sizer = wx.GridBagSizer(5, 5)

        # Stock tickers
        for i in range(4):
            stock_label = wx.StaticText(self, label=f"Stock {chr(65 + i)} Ticker:")
            self.stock_inputs.append(wx.TextCtrl(self, size=(80, -1)))
            main_sizer.Add(stock_label, pos=(0, i * 2))
            main_sizer.Add(self.stock_inputs[-1], pos=(0, i * 2 + 1))

        # Weights
        for i in range(4):
            weight_label = wx.StaticText(self, label=f"Stock {chr(65 + i)} Weight:")
            self.weight_inputs.append(wx.TextCtrl(self, size=(80, -1)))
            main_sizer.Add(weight_label, pos=(1, i * 2))
            main_sizer.Add(self.weight_inputs[-1], pos=(1, i * 2 + 1))

        # Date and benchmark
        date_label = wx.StaticText(self, label="Start Date (dd/mm/yyyy):")
        self.date_input = wx.TextCtrl(self, size=(100, -1))
        benchmark_label = wx.StaticText(self, label="Benchmark Ticker:")
        self.benchmark_input = wx.TextCtrl(self, size=(100, -1))

        main_sizer.Add(date_label, pos=(2, 0))
        main_sizer.Add(self.date_input, pos=(2, 1))
        main_sizer.Add(benchmark_label, pos=(2, 2))
        main_sizer.Add(self.benchmark_input, pos=(2, 3))

        # ADD CONFIDENCE INTERVAL SELECTION HERE (NEW CODE)
        ci_label = wx.StaticText(self, label="Confidence Level (%):")
        self.ci_choices = ['90', '95', '99']
        self.ci_combo = wx.ComboBox(self, choices=self.ci_choices, value='95', style=wx.CB_READONLY)

        main_sizer.Add(ci_label, pos=(2, 4))
        main_sizer.Add(self.ci_combo, pos=(2, 5))

        # Retrieve button
        self.retrieve_btn = wx.Button(self, label="Retrieve Data")
        self.retrieve_btn.Bind(wx.EVT_BUTTON, self.on_retrieve)
        main_sizer.Add(self.retrieve_btn, pos=(3, 0), span=(1, 4), flag=wx.EXPAND | wx.ALL, border=5)

        # Warning label
        self.warning_label = wx.StaticText(self, label="")
        main_sizer.Add(self.warning_label, pos=(4, 0), span=(1, 4), flag=wx.EXPAND | wx.ALL, border=5)

        self.SetSizer(main_sizer)

    def on_retrieve(self, event):
        try:
            # Get confidence interval
            confidence_level = float(self.ci_combo.GetValue()) / 100
            alpha = 1 - confidence_level

            # Validate inputs
            stocks = [inp.GetValue().upper().strip() for inp in self.stock_inputs]
            weights = [inp.GetValue().strip() for inp in self.weight_inputs]
            date_str = self.date_input.GetValue().strip()
            benchmark = self.benchmark_input.GetValue().upper().strip()

            # Check empty fields
            if any(not s for s in stocks) or any(not w for w in weights) or not date_str or not benchmark:
                raise ValueError("All fields must be filled")

            # Validate weights
            try:
                weights = list(map(float, weights))
                if not math.isclose(sum(weights), 1.0, rel_tol=1e-4):
                    raise ValueError("Weights must sum to 1.0")
            except ValueError:
                raise ValueError("Weights must be numeric values")

            # Validate date
            start_date = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()

            # Download data
            stock_data = yf.download(stocks, start=start_date)['Close']
            benchmark_data = yf.download(benchmark, start=start_date)['Close']

            # Combine data
            combined = pd.concat([stock_data, benchmark_data], axis=1)
            combined.columns = stocks + [benchmark]
            combined = combined.dropna()

            data = combined[stocks]
            benchmark_series = combined[benchmark]

            # Calculate portfolio (CRITICAL FIX HERE)
            returns = data.pct_change().dropna()
            positions = pd.DataFrame(
                {stock: [weight] for stock, weight in zip(stocks, weights)},  # Fixed parenthesis
                index=[returns.index[0]]
            ).reindex(returns.index).ffill()

            portfolio = positions.shift() * (1 + returns).cumprod(axis=0)
            portfolio['total_wealth'] = portfolio.sum(axis=1)
            portfolio['returns'] = portfolio['total_wealth'].pct_change()

            # Publish data
            pub.sendMessage("data_updated",
                            data=data,
                            weights=weights,
                            stocks=stocks,
                            portfolio=portfolio,
                            benchmark=benchmark_series,
                            alpha=alpha)

            self.warning_label.SetLabel("Data retrieved successfully")
            self.warning_label.SetForegroundColour(wx.Colour(0, 100, 0))

        except Exception as e:
            self.warning_label.SetLabel(str(e))
            self.warning_label.SetForegroundColour(wx.RED)
            wx.Bell()



class PageOne(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        pub.subscribe(self.update_data, "data_updated")

        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def update_data(self, data, weights, stocks, portfolio, benchmark, alpha):
        try:
            self.ax1.clear()
            self.ax2.clear()

            # Plot histograms
            returns = data.pct_change().dropna()
            for stock in stocks:
                self.ax1.hist(returns[stock], bins=50, alpha=0.5, label=stock)
            self.ax1.legend()
            self.ax1.set_title("Return Distributions")

            # Plot cumulative returns
            cum_returns = (1 + returns).cumprod()
            for stock in stocks:
                self.ax2.plot(cum_returns[stock], label=stock)
            self.ax2.plot((1 + portfolio['returns']).cumprod(), label='Portfolio', lw=2)
            self.ax2.legend()
            self.ax2.set_title("Cumulative Returns")

            self.canvas.draw()

        except Exception as e:
            print(f"Error updating plots: {e}")


class PageTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        pub.subscribe(self.update_data, "data_updated")

        self.text_output = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text_output, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def update_data(self, data, weights, stocks, portfolio, benchmark, alpha):
        try:
            returns = data.pct_change().dropna()
            portfolio_returns = portfolio['returns'].dropna()
            benchmark_returns = benchmark.pct_change().dropna()

            # Calculate VaRs with dynamic alpha
            output = []
            output.append(f"Historical VaR ({1-alpha:.0%}):")
            for stock in stocks:
                var = returns[stock].quantile(alpha) * 100
                output.append(f"{stock}: {var:.2f}%")

            output.append(f"\nParametric VaR ({1-alpha:.0%}):")
            for stock in stocks:
                mu = returns[stock].mean()
                sigma = returns[stock].std()
                var = scipy.stats.norm.ppf(alpha, mu, sigma) * 100
                output.append(f"{stock}: {var:.2f}%")

            output.append("\nPortfolio VaR:")
            output.append(f"Historical: {portfolio_returns.quantile(alpha)*100:.2f}%")
            mu_p = portfolio_returns.mean()
            sigma_p = portfolio_returns.std()
            var_p = scipy.stats.norm.ppf(alpha, mu_p, sigma_p) * 100
            output.append(f"Parametric: {var_p:.2f}%")

            self.text_output.SetValue("\n".join(output))

        except Exception as e:
            self.text_output.SetValue(f"Error calculating VARs: {str(e)}")


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Portfolio Analysis Tool", size=(1000, 800))

        main_panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add input panel
        self.input_panel = InputPanel(main_panel)
        sizer.Add(self.input_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Create notebook
        notebook = wx.Notebook(main_panel)

        # Add pages
        page1 = PageOne(notebook)
        page2 = PageTwo(notebook)
        notebook.AddPage(page1, "Descriptive Data")
        notebook.AddPage(page2, "Risk Analysis")

        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        main_panel.SetSizer(sizer)

        self.Centre()
        self.Show()


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()