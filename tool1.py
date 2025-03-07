import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
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

        # Date and parameters
        date_label = wx.StaticText(self, label="Start Date (dd/mm/yyyy):")
        self.date_input = wx.TextCtrl(self, size=(100, -1))
        sim_label = wx.StaticText(self, label="Simulations:")
        self.sim_input = wx.TextCtrl(self, size=(100, -1), value="10000")
        ci_label = wx.StaticText(self, label="Confidence Level (%):")
        self.ci_choices = ['90', '95', '99']
        self.ci_combo = wx.ComboBox(self, choices=self.ci_choices, value='95', style=wx.CB_READONLY)

        main_sizer.Add(date_label, pos=(2, 0))
        main_sizer.Add(self.date_input, pos=(2, 1))
        main_sizer.Add(sim_label, pos=(2, 2))
        main_sizer.Add(self.sim_input, pos=(2, 3))
        main_sizer.Add(ci_label, pos=(2, 4))
        main_sizer.Add(self.ci_combo, pos=(2, 5))

        # Retrieve button
        self.retrieve_btn = wx.Button(self, label="Retrieve Data")
        self.retrieve_btn.Bind(wx.EVT_BUTTON, self.on_retrieve)
        main_sizer.Add(self.retrieve_btn, pos=(3, 0), span=(1, 6), flag=wx.EXPAND | wx.ALL, border=5)

        # Warning label
        self.warning_label = wx.StaticText(self, label="")
        main_sizer.Add(self.warning_label, pos=(4, 0), span=(1, 6), flag=wx.EXPAND | wx.ALL, border=5)

        self.SetSizer(main_sizer)

    def on_retrieve(self, event):
        try:
            # Get inputs
            confidence_level = float(self.ci_combo.GetValue()) / 100
            alpha = 1 - confidence_level
            n_simulations = int(self.sim_input.GetValue())

            # Validate inputs
            stocks = [inp.GetValue().upper().strip() for inp in self.stock_inputs]
            weights = [float(inp.GetValue().strip()) for inp in self.weight_inputs]
            date_str = self.date_input.GetValue().strip()

            if not all(stocks) or not all(weights) or not date_str:
                raise ValueError("All fields must be filled")
            if not math.isclose(sum(weights), 1.0, rel_tol=1e-4):
                raise ValueError("Weights must sum to 1.0")

            # Download data
            start_date = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
            stock_data = yf.download(stocks, start=start_date)['Close']
            combined = stock_data.dropna()

            # Process data
            returns = combined.pct_change().dropna()

            # Calculate parameters for Monte Carlo
            mu = returns.mean().values
            cov_matrix = returns.cov().values
            weights = np.array(weights)

            # Publish data
            pub.sendMessage("data_updated",
                            data=combined,
                            weights=weights,
                            stocks=stocks,
                            mu=mu,
                            cov_matrix=cov_matrix,
                            n_simulations=n_simulations,
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

    def update_data(self, data, weights, stocks, mu, cov_matrix, n_simulations, alpha):
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
        self.ax2.legend()
        self.ax2.set_title("Cumulative Returns")

        self.canvas.draw()


class PageTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        pub.subscribe(self.update_data, "data_updated")

        self.text_output = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text_output, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def update_data(self, data, weights, stocks, mu, cov_matrix, n_simulations, alpha):
        try:
            # Monte Carlo simulation
            L = np.linalg.cholesky(cov_matrix)
            dt = 1  # Daily horizon

            # Generate correlated random shocks
            Z = np.random.normal(size=(n_simulations, len(stocks)))
            correlated_returns = mu * dt + np.dot(Z, L.T) * np.sqrt(dt)

            # Calculate individual stock VaRs
            individual_vars = {}
            for i, stock in enumerate(stocks):
                stock_losses = -correlated_returns[:, i]
                individual_vars[stock] = np.percentile(stock_losses, 100 * (1 - alpha))

            # Calculate portfolio VaR
            portfolio_returns = np.dot(correlated_returns, weights)
            portfolio_losses = -portfolio_returns
            portfolio_var = np.percentile(portfolio_losses, 100 * (1 - alpha))

            # Prepare output
            output = [
                f"Monte Carlo VaR Results ({1 - alpha:.0%} Confidence):",
                f"Simulations: {n_simulations:,}",
                f"Time Horizon: {dt} day(s)\n",
                "Individual VaRs:"
            ]

            for stock, var in individual_vars.items():
                output.append(f"  {stock}: {var * 100:.2f}%")

            output.extend([
                f"\nPortfolio VaR: {portfolio_var * 100:.2f}%",
                f"\nWeights: {dict(zip(stocks, weights))}"
            ])

            self.text_output.SetValue("\n".join(output))

        except Exception as e:
            self.text_output.SetValue(f"Error in simulation:\n{str(e)}")


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Portfolio VaR Analyzer", size=(1200, 800))
        main_panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.input_panel = InputPanel(main_panel)
        sizer.Add(self.input_panel, 0, wx.EXPAND | wx.ALL, 5)

        notebook = wx.Notebook(main_panel)
        notebook.AddPage(PageOne(notebook), "Descriptive Analysis")
        notebook.AddPage(PageTwo(notebook), "Monte Carlo VaR")
        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)

        main_panel.SetSizer(sizer)
        self.Centre()
        self.Show()


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()