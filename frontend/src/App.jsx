import { useEffect, useState } from "react";
import API from "./api";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  BarChart,
  Bar
} from "recharts";

import {
  FaRobot,
  FaChartLine,
  FaDatabase,
  FaHome,
  FaUpload,
  FaDownload,
  FaTable,
  FaPaperPlane,
  FaInfoCircle
} from "react-icons/fa";

import "./index.css";

const PERIODS = ["1M", "3M", "6M", "1Y", "ALL"];

export default function App() {

  const [companies, setCompanies] = useState([]);

  const [selectedCompany, setSelectedCompany] = useState("");
  
  const [companyProfile, setCompanyProfile] = useState(null);

  const [showFullProfile, setShowFullProfile] = useState(false);

  const [page, setPage] = useState("Home");

  const [period, setPeriod] = useState("ALL");

  const [dashboard, setDashboard] = useState(null);

  const [stockInfo, setStockInfo] = useState(null);

  const [analytics, setAnalytics] = useState(null);

  const [prediction, setPrediction] = useState(null);

  const [predictionLoading, setPredictionLoading] = useState(false);

  const [predictionHistory, setPredictionHistory] = useState([]);

  const [forecastLock, setForecastLock] = useState(false);

  const [showHistory, setShowHistory] = useState(false);

  const [stockSuggestions, setStockSuggestions] = useState([]);

  const [selectedSuggestion, setSelectedSuggestion] = useState(null);

  const [marketSummary, setMarketSummary] = useState([]);

  const [tableData, setTableData] = useState([]);

  const [priceView, setPriceView] = useState("All");

  const [chatOpen, setChatOpen] = useState(false);

  const [chatQuestion, setChatQuestion] = useState("");

  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Hello, I am your RSEB Financial Intelligence Assistant."
    }
  ]);

  useEffect(() => {

    API.get("/companies").then((res) => {

      setCompanies(res.data.companies || []);

      if (res.data.companies?.length) {
        setSelectedCompany(res.data.companies[0]);
      }

      loadStockSuggestions();

    });

  }, []);

  useEffect(() => {

    if (!selectedCompany) return;

    loadCompanyData(selectedCompany, period);

  }, [selectedCompany, period]);

  const loadCompanyData = async (company, selectedPeriod) => {
  setPrediction(null);
  setPredictionHistory([]);

  try {

      const dash = await API.get(
  `/dashboard/${company}?period=${selectedPeriod}`
);

      const ana = await API.get(
        `/analytics/${company}?period=${selectedPeriod}`
      );

      const profile = await API.get(`/company-profile/${company}`);
      setCompanyProfile(
        profile.data.extra_info || null
      );

setShowFullProfile(false);
      setShowFullProfile(false);

      setDashboard(dash.data);

      setAnalytics(ana.data);

      const stockRes = await API.get(`/rseb-stock-info/${company}`);
      setStockInfo(stockRes.data.metrics || null);

      const history = await API.get(
        `/prediction-history?company=${company}`
      );

      setPredictionHistory(history.data.history || []);

      const market = await API.get(
        `/market-summary?period=${selectedPeriod}`
      );

      setMarketSummary(market.data.companies || []);

    } catch (err) {

      console.error(err);

    }

  };

  const loadStockSuggestions = async () => {
  try {
    const res = await API.get("/recommend/demo_user?top_k=5");

    console.log("Recommendation response:", res.data);

    if (Array.isArray(res.data?.recommendations)) {
      setStockSuggestions(res.data.recommendations);
    } else {
      setStockSuggestions([]);
    }

  } catch (err) {
    console.error("Recommendation error:", err);
    setStockSuggestions([]);
  }
};

  const runForecast = async () => {
  if (forecastLock || predictionLoading) return;

  const companyToPredict = selectedCompany;

  if (!companyToPredict) return;

  setForecastLock(true);
  setPredictionLoading(true);
  setPrediction(null);

  try {
    const pred = await API.get(
      `/predict/${companyToPredict}?scrape_latest=false&headless=true`
    );

    const forecastData = pred.data.data;

    setPrediction({
      ...forecastData,
      company: companyToPredict.toUpperCase()
    });

    const history = await API.get(
      `/prediction-history?company=${companyToPredict}`
    );

    setPredictionHistory(history.data.history || []);
  } catch (err) {
    console.error("Forecasting error:", err);
    setPrediction(null);
  } finally {
    setPredictionLoading(false);

    setTimeout(() => {
      setForecastLock(false);
    }, 1000);
  }
};
  const sendFinancialChat = async () => {
  if (!chatQuestion.trim()) return;

  const userText = chatQuestion.trim();

  setChatMessages((prev) => [
    ...prev,
    {
      role: "user",
      text: userText
    }
  ]);

  setChatQuestion("");

  try {
    const res = await API.post("/financial-agent/chat", {
      question: userText,
      company: dashboard?.company
    });

    const rawTrendData =
  res.data.trend_data ||
  res.data.data?.trend ||
  res.data.data?.trend?.data ||
  [];

const normalizedTrendData = Array.isArray(rawTrendData)
  ? rawTrendData.map((row) => ({
      year: row.year || row.fiscal_year || row.Year,
      value: Number(row.value || row.Value || row.amount)
    }))
  : [];

setChatMessages((prev) => [
  ...prev,
  {
    role: "assistant",
    text: res.data.answer,
    explanation: res.data.explanation,
    type: res.data.type,
    trendData: normalizedTrendData
  }
]);
  } catch (err) {
    console.error(err);

    setChatMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        text: "Sorry, something went wrong while processing your question."
      }
    ]);
  }
};

  if (!dashboard) {
    return <div className="loading">Loading...</div>;
  }

  const chartData = (dashboard.chart_data || []).slice(-300);

  const trend = dashboard.metrics?.trend || "N/A";

  const activePrediction =
  prediction &&
  String(prediction.company).toUpperCase() === String(selectedCompany).toUpperCase()
    ? prediction
    : null;

  const buildForecastChartData = () => {
  const historical = chartData.map((row) => ({
    Date: row.Date,
    Close: Number(row.Close),
    sevenDayForecast: null,
    thirtyDayForecast: null
  }));

  if (!activePrediction || historical.length === 0) {
    return historical;
  }

  const lastRow = historical[historical.length - 1];
  const lastDate = new Date(lastRow.Date);
  const lastClose = Number(lastRow.Close);

  const day7 = new Date(lastDate);
  day7.setDate(day7.getDate() + 7);

  const day30 = new Date(lastDate);
  day30.setDate(day30.getDate() + 30);

  return [
    ...historical,
    {
      Date: day7.toISOString().split("T")[0],
      Close: null,
      sevenDayForecast: Number(activePrediction.predicted_week_price),
      thirtyDayForecast: null
    },
    {
      Date: day30.toISOString().split("T")[0],
      Close: null,
      sevenDayForecast: null,
      thirtyDayForecast: Number(activePrediction.predicted_month_price)
    }
  ];
};

const forecastChartData = buildForecastChartData();


  return (

    <div className="app-container">

      {/* ================= NAVBAR ================= */}

      <header className="topbar">

        <div className="logo-section">

          <div className="brand-text">
            <h1>RSEB FINANCEHUB</h1>
            <p>Bhutan Stock Intelligence Platform</p>
          </div>

        </div>

        <nav className="topnav">

          <button
            className={page === "Home" ? "active-nav" : ""}
            onClick={() => setPage("Home")}
          >
            <FaHome />
            Home
          </button>

          <button
            className={page === "Forecasting" ? "active-nav" : ""}
            onClick={() => setPage("Forecasting")}
          >
            <FaChartLine />
            Forecasting
          </button>

          <button
            className={page === "About" ? "active-nav" : ""}
            onClick={() => setPage("About")}
          >
            <FaInfoCircle />
            About
          </button>

        </nav>

        <div className="rseb-right-logo">
          <img src="/logos/RSEB.png" alt="RSEB" />
        </div>

      </header>

      {/* ================= HOME PAGE ================= */}

      {page === "Home" && (

        <main className="home-layout">

          {/* LEFT PANEL */}

          <section className="left-panel glass-card">

            <div className="panel-header">
              <h2>Featured Companies</h2>
            </div>

            <div className="company-list">

              {marketSummary.map((item) => (

                <button
                  key={item.company}
                  className={`company-item ${
                    selectedCompany === item.company
                      ? "selected-company"
                      : ""
                  }`}
                  onClick={() => setSelectedCompany(item.company)}
                >

                  <div className="company-left">

                    <img
                      src={`/logos/${item.company}.png`}
                      alt={item.company}
                      className="company-logo"
                    />

                    <div>
                      <h4>{item.company}</h4>
                      <p>{item.trend}</p>
                    </div>

                  </div>

                  <strong
                    className={
                      Number(item.total_return_pct) >= 0
                        ? "green"
                        : "red"
                    }
                  >
                    {item.total_return_pct}%
                  </strong>

                </button>

              ))}

            </div>

          </section>

          {/* CENTER PANEL */}

          <section className="center-panel glass-card">

            <div className="overview-top">

              <div>

                <span className="overview-label">
                  Company Overview
                </span>

                <h1>{dashboard.company}</h1>

                {companyProfile ? (
  <div className="company-info-clean">
    <p><span>Established:</span> {companyProfile["Established"]}</p>
    <p><span>Sector:</span> {companyProfile["Sector"]}</p>
    <p><span>Address:</span> {companyProfile["Address"]}</p>
    <p><span>Paid-up Shares:</span> {companyProfile["Paid-up Shares"]}</p>
    <p>
      <span>Website:</span>{" "}
      {companyProfile["Website"] !== "N/A" ? (
        <a href={companyProfile["Website"]} target="_blank" rel="noreferrer">
          {companyProfile["Website"]}
        </a>
      ) : (
        "N/A"
      )}
    </p>
  </div>
) : (
  <p>Company information loading...</p>
)}

{companyProfile?.description?.length > 280 && (
  <button
    className="show-more-btn"
    onClick={() => setShowFullProfile(!showFullProfile)}
  >
    {showFullProfile ? "Show less" : "Show more"}
  </button>
)}

              </div>

              <img
                src={`/logos/${dashboard.company}.png`}
                alt={dashboard.company}
                className="overview-logo"
              />

            </div>

            <div className="metrics-row">

              <div className="metric-box">
                <span>Latest Close</span>
                <h2>{dashboard.metrics.latest_close}</h2>
              </div>

              <div className="metric-box">
                <span>Return</span>
                <h2>{dashboard.metrics.return}</h2>
              </div>

              <div className="metric-box">
                <span>Volatility</span>
                <h2>{dashboard.metrics.volatility}</h2>
              </div>

              <div className="metric-box">
                <span>Trend</span>
                <h2>{trend}</h2>
              </div>

            </div>
                        <div className="period-tabs">

              {PERIODS.map((p) => (

                <button
                  key={p}
                  className={period === p ? "active-period" : ""}
                  onClick={() => setPeriod(p)}
                >
                  {p}
                </button>

              ))}

            </div>

            <div className="chart-card">

              <div className="chart-header">

                <div>
                  <h3>Stock Price Movement</h3>
                  <p>Close price trend with SMA and EMA indicators</p>
                </div>

                <select
                  value={priceView}
                  onChange={(e) => setPriceView(e.target.value)}
                >
                  <option>All</option>
                  <option>Close</option>
                  <option>SMA</option>
                  <option>EMA</option>
                </select>

              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Date" />
                  <YAxis />
                  <Tooltip />

                  {(priceView === "All" || priceView === "Close") && (
                    <Line
                      type="monotone"
                      dataKey="Close"
                      stroke="#1d4ed8"
                      strokeWidth={3}
                      dot={false}
                    />
                  )}

                  {(priceView === "All" || priceView === "SMA") && (
                    <Line
                      type="monotone"
                      dataKey="sma_5"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={false}
                    />
                  )}

                  {(priceView === "All" || priceView === "EMA") && (
                    <Line
                      type="monotone"
                      dataKey="ema_5"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={false}
                    />
                  )}

                </LineChart>
              </ResponsiveContainer>

              {stockInfo && (
  <div className="stock-extra-info">

    <div className="info-grid">

      <div><span>Open</span><strong>{stockInfo["Open"] || "N/A"}</strong></div>

      <div><span>Vol</span><strong>{stockInfo["Vol"] || "N/A"}</strong></div>

      <div><span>52W H</span><strong>{stockInfo["52W H"] || "N/A"}</strong></div>

      <div><span>Div yield</span><strong>{stockInfo["Div yield"] || "N/A"}</strong></div>

      <div><span>High</span><strong>{stockInfo["High"] || "N/A"}</strong></div>

      <div><span>P/E</span><strong>{stockInfo["P/E"] || "N/A"}</strong></div>

      <div><span>52W L</span><strong>{stockInfo["52W L"] || "N/A"}</strong></div>

      <div><span>Book Val</span><strong>{stockInfo["Book Val"] || "N/A"}</strong></div>

      <div><span>Low</span><strong>{stockInfo["Low"] || "N/A"}</strong></div>

      <div><span>Mkt cap</span><strong>{stockInfo["Mkt cap"] || "N/A"}</strong></div>

      <div><span>Avg Vol</span><strong>{stockInfo["Avg Vol"] || "N/A"}</strong></div>

      <div><span>EPS</span><strong>{stockInfo["EPS"] || "N/A"}</strong></div>

    </div>

  </div>
)}

            </div>

          </section>

          {/* RIGHT PANEL */}

          <section className="right-panel glass-card">
  <div className="panel-header">
    <h2>Recommended Stocks</h2>
  </div>

 <div className="suggestion-list">
  {stockSuggestions.length > 0 ? (
    stockSuggestions.map((item, index) => {
      const company =
        item.recommended_company ||
        item.forecast?.company ||
        "N/A";

      const action =
        item.forecast?.predicted_direction ||
        item.reason?.predicted_direction ||
        "HOLD";

      const score =
        item.final_score ??
        item.reason?.direction_confidence ??
        null;

      const weekChange =
        item.forecast?.week_change_percent ??
        item.reason?.week_change_percent ??
        null;

      return (
        <div className="suggestion-card" key={index}>
          <div className="suggestion-main">
            <img
              src={`/logos/${company}.png`}
              alt={company}
              onError={(e) => {
                e.currentTarget.src = "/logos/RSEB.png";
              }}
            />

            <div className="suggestion-info">
              <h4>{company}</h4>

              <p className={`action-text ${action.toLowerCase()}`}>
                {action}
              </p>

              <small>AI Insight</small>
            </div>

            {weekChange !== null && (
              <span
                className={`change-pill ${
                  Number(weekChange) >= 0
                    ? "positive-pill"
                    : "negative-pill"
                }`}
              >
                {Number(weekChange).toFixed(2)}%
              </span>
            )}
          </div>

          <div className="suggestion-footer">
            <span>
              Score: {score !== null ? Number(score).toFixed(2) : "N/A"}
            </span>

            <button
              className="view-details-btn"
              onClick={() => setSelectedSuggestion(item)}
            >
              View Details
            </button>
          </div>
        </div>
      );
    })
  ) : (
    <div className="empty-suggestion">
      <img src="/logos/RSEB.png" alt="RSEB" />
      <p>No recommendation available yet.</p>
    </div>
  )}
</div>
</section>

        </main>

      )}

      {/* ================= FORECASTING PAGE ================= */}

      {page === "Forecasting" && (

        <main className="forecast-layout">

          <section className="forecast-hero glass-card">

            <div>
              <span className="overview-label">
                Forecasting & Analysis
              </span>

              <h1>Predictive Forecasting Center</h1>

              <p>
                Generate AI-based short-term and medium-term
                forecasts using the latest RSEB stock data.
              </p>
              <div className="forecast-company-select">
  <label>Select Company</label>

  <select
  value={selectedCompany}
  onChange={(e) => {
    const newCompany = e.target.value;

    setSelectedCompany(newCompany);
    setPrediction(null);
    setPredictionHistory([]);
    setPredictionLoading(false);
  }}
>
    {companies.map((company) => (
      <option key={company} value={company}>
        {company}
      </option>
    ))}
  </select>
</div>


            </div>

            <button
  className="run-btn"
  onClick={runForecast}
  disabled={predictionLoading || forecastLock}
>
  {predictionLoading ? "Running..." : "Run Forecast"}
</button>

          </section>

          <section className="glass-card chart-card">

            <div className="chart-header">
              <div>
                <h3>Forecast Price Movement</h3>
                <p>Historical price movement with AI prediction results</p>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={forecastChartData}>
  <CartesianGrid strokeDasharray="3 3" />

  <XAxis
  dataKey="Date"
  minTickGap={40}
  interval="preserveStartEnd"
  tickFormatter={(value) => {
    const date = new Date(value);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short"
    });
  }}
/>

  <YAxis />

  <Tooltip />

  <Legend />

  {/* Historical Price */}

  <Line
    type="monotone"
    dataKey="Close"
    name="Historical Close"
    stroke="#1d4ed8"
    strokeWidth={3}
    dot={false}
    connectNulls={false}
  />

  {/* 7 Day Forecast */}

 <Line
  type="monotone"
  dataKey="sevenDayForecast"
  name="7-Day Forecast"
  stroke="#16a34a"
  strokeWidth={3}
  strokeDasharray="8 6"
  dot={{ r: 5, fill: "#16a34a" }}
  connectNulls={true}
/>

<Line
  type="monotone"
  dataKey="thirtyDayForecast"
  name="30-Day Forecast"
  stroke="#dc2626"
  strokeWidth={3}
  strokeDasharray="8 6"
  dot={{ r: 5, fill: "#dc2626" }}
  connectNulls={true}
/>

</LineChart>
            </ResponsiveContainer>

          </section>

          <section className="forecast-grid">

  <div className="forecast-card">
    <span>Current Price</span>
    <h2>
      {activePrediction?.current_price !== undefined
        ? `Nu. ${activePrediction.current_price}`
        : "N/A"}
    </h2>
  </div>

  <div className="forecast-card">
    <span>1 Week Forecast</span>
    <h2>
      {activePrediction?.predicted_week_price !== undefined
        ? `Nu. ${activePrediction.predicted_week_price}`
        : "N/A"}
    </h2>
    <p
  className={
    Number(activePrediction?.week_change_percent) > 0
      ? "forecast-positive"
      : Number(activePrediction?.week_change_percent) < 0
      ? "forecast-negative"
      : "forecast-neutral"
  }
>
  {activePrediction?.week_change_percent !== undefined
    ? `${activePrediction.week_change_percent}%`
    : "N/A"}
</p>
  </div>

  <div className="forecast-card">
    <span>1 Month Forecast</span>
    <h2>
      {activePrediction?.predicted_month_price !== undefined
        ? `Nu. ${activePrediction.predicted_month_price}`
        : "N/A"}
    </h2>
    <p
  className={
    Number(activePrediction?.month_change_percent) > 0
      ? "forecast-positive"
      : Number(activePrediction?.month_change_percent) < 0
      ? "forecast-negative"
      : "forecast-neutral"
  }
>
      {activePrediction?.month_change_percent !== undefined
        ? `${activePrediction.month_change_percent}%`
        : "N/A"}
    </p>
  </div>

  <div className="forecast-card">
    <span>Direction</span>
    <h2>{activePrediction?.predicted_direction || "N/A"}</h2>
    <p>
      {activePrediction?.direction_confidence !== undefined
        ? `Confidence: ${activePrediction.direction_confidence}%`
        : "N/A"}
    </p>
  </div>

</section>

{activePrediction?.forecast_insight && (
  <section className="glass-card forecast-insight-card">
    <h3>AI Forecast Explanation</h3>
    <p>{activePrediction.forecast_insight}</p>
  </section>
)}

          <section className="glass-card table-card">

  <div className="panel-header history-header">
    <h2>Prediction History</h2>

    <button
      className="view-details-btn"
      onClick={() => setShowHistory(!showHistory)}
    >
      {showHistory ? "Hide History" : "View History"}
    </button>
  </div>

  {showHistory && (
    <div className="table-wrap">

      <table>
        <thead>
          <tr>
            <th>Saved At</th>
            <th>Company</th>
            <th>Model</th>
            <th>Latest Close</th>
            <th>1 Week</th>
            <th>Direction</th>
            <th>1 Month</th>
            <th>Direction</th>
          </tr>
        </thead>

        <tbody>
          {predictionHistory.map((row, index) => (
            <tr key={index}>
              <td>{row.saved_at || row.timestamp || "N/A"}</td>
              <td>{row.company || "N/A"}</td>
              <td>{row.model_used || "TiDE + XGBoost"}</td>
              <td>{row.current_price || row.latest_close || "N/A"}</td>
              <td>
                {row.predicted_week_price ||
                  row.one_week?.predicted_close ||
                  "N/A"}
              </td>
              <td>
                {row.predicted_direction ||
                  row.one_week?.direction ||
                  "N/A"}
              </td>
              <td>
                {row.predicted_month_price ||
                  row.one_month?.predicted_close ||
                  "N/A"}
              </td>
              <td>
                {row.predicted_direction ||
                  row.one_month?.direction ||
                  "N/A"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

    </div>
  )}

</section>

        </main>

      )}

      {/* ================= ABOUT PAGE ================= */}

      {page === "About" && (

        <main className="about-layout">

          <section className="about-hero glass-card">

            <span className="overview-label">
              About The Project
            </span>

            <h1>AI-Driven Stock Price Prediction, Recommendation and Investor Assistant</h1>

            <p>
              This platform is designed to support Bhutanese investors through
              stock analytics, AI forecasting, personalized recommendations,
              and financial statement-based chatbot assistance.
            </p>

          </section>

          <section className="about-grid">

            <div className="glass-card about-box">
              <h3>Project Overview</h3>
              <p>
                The system combines market data, financial records, forecasting
                models, recommendation logic, and an LLM-based assistant to make
                RSEB stock information easier to understand.
              </p>
            </div>

            <div className="glass-card about-box">
              <h3>Mission</h3>
              <p>
                To improve financial transparency, investor confidence, and
                data-driven decision making in Bhutan’s capital market.
              </p>
            </div>

            <div className="glass-card about-box">
              <h3>Key Contributors</h3>
              <p>
                Developed as a capstone project with guidance from academic
                supervisors and support from the RSEB IT Division.
              </p>
            </div>

            <div className="glass-card about-box">
              <h3>Core Technology Stack</h3>
              <p>
                React, FastAPI, Python, machine learning forecasting models,
                Recharts, financial data processing, and LLM-powered insights.
              </p>
            </div>

          </section>

        </main>

      )}

      {/* ================= CHATBOT ================= */}

      <button
        className="chatbot-button"
        onClick={() => setChatOpen(true)}
      >
        <FaRobot />
      </button>

      {chatOpen && (

        <div className="chat-window">

          <div className="chat-header">
            <h3>RSEB Assistant</h3>
            <button onClick={() => setChatOpen(false)}>×</button>
          </div>

          <div className="chat-body">

            {chatMessages.map((msg, index) => (
  <div
    key={index}
    className={`chat-message ${msg.role}`}
  >
    <div>{msg.text}</div>

    {msg.explanation && (
      <div className="chat-explanation">
        <strong>AI Explanation</strong>
        <p>{msg.explanation}</p>
      </div>
    )}


  </div>
))}

          </div>

          <div className="chat-input">

            <input
              value={chatQuestion}
              onChange={(e) => setChatQuestion(e.target.value)}
              placeholder="Ask about revenue, profit, ratios..."
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  sendFinancialChat();
                }
              }}
            />

            <button onClick={sendFinancialChat}>
              <FaPaperPlane />
            </button>

          </div>

        </div>

      )}
      
      {selectedSuggestion && (
  <div className="modal-overlay">
    <div className="ai-modal">

      <button
        className="modal-close"
        onClick={() => setSelectedSuggestion(null)}
      >
        ×
      </button>

      <h2>
        {selectedSuggestion.recommended_company}
      </h2>

      <p>
        This stock recommendation is generated using
        AI-based forecasting, trend analysis,
        volatility behaviour, and prediction confidence.
      </p>

      <div className="modal-stats">

        <div>
          <span>Direction</span>
          <strong>
            {
              selectedSuggestion.reason
                ?.predicted_direction || "N/A"
            }
          </strong>
        </div>

        <div>
          <span>Confidence</span>
          <strong>
            {
              selectedSuggestion.reason
                ?.direction_confidence
                  ? Number(
                      selectedSuggestion.reason
                        ?.direction_confidence
                    ).toFixed(2)
                  : "N/A"
            }%
          </strong>
        </div>

        <div>
          <span>Week Change</span>
          <strong>
            {
              selectedSuggestion.reason
                ?.week_change_percent
                  ? Number(
                      selectedSuggestion.reason
                        ?.week_change_percent
                    ).toFixed(2)
                  : "N/A"
            }%
          </strong>
        </div>

        <div>
          <span>Month Change</span>
          <strong>
            {
              selectedSuggestion.reason
                ?.month_change_percent
                  ? Number(
                      selectedSuggestion.reason
                        ?.month_change_percent
                    ).toFixed(2)
                  : "N/A"
            }%
          </strong>
        </div>

      </div>

      <p className="ai-note">
        AI Insight: {
          (() => {
            const direction = selectedSuggestion.reason?.predicted_direction;
            const confidence = Number(selectedSuggestion.reason?.direction_confidence || 0);
            const week = Number(selectedSuggestion.reason?.week_change_percent || 0);
            const month = Number(selectedSuggestion.reason?.month_change_percent || 0);
            const company = selectedSuggestion.recommended_company;

            if (direction === "UP") {
              if (confidence > 80) {
                return `${company} demonstrates strong bullish momentum with high AI confidence and positive projected growth in upcoming trading sessions.`;
              }
              return `${company} is showing moderate upward movement supported by improving short-term market sentiment and forecast trends.`;
            }

            if (direction === "DOWN") {
              if (month < 0) {
                return `${company} reflects declining market behaviour with weaker future outlook and reduced momentum across recent trading periods.`;
              }
              return `${company} indicates cautious trading activity despite short-term market fluctuations and lower directional confidence.`;
            }

            if (Math.abs(week) < 0.5 && Math.abs(month) < 0.5) {
              return `${company} is maintaining relatively stable price behaviour with low volatility and balanced investor activity.`;
            }

            return `${company} is showing no strong directional signal at this time.`;
          })()
        }
      </p>

    </div>
  </div>
)}

    </div>

  );

}