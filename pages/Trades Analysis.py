import os
import glob
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
import plotly.express as px

def find_default_csv_file(base_dir: str) -> str | None:
    backtest_dir = os.path.join(base_dir, "ListofTrades")
    pattern = os.path.join(backtest_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 2021-03-25 08:00
    df["Datetime"] = pd.to_datetime(df["Date/Time"], format="%Y-%m-%d %H:%M")
    return df

def nested_pie_chart(df: pd.DataFrame) -> go.Figure:
    # 取 Entry / Exit，並按 Trade # 排序確保正確配對
    entries = df[df["Type"].str.contains("Entry")].copy()
    exits = df[df["Type"].str.contains("Exit")].copy()
    
    # 按 Trade # 排序
    entries = entries.sort_values("Trade #")
    exits = exits.sort_values("Trade #")
    
    # 確保每個 Trade # 都有對應的 Entry 和 Exit
    common_trades = set(entries["Trade #"]) & set(exits["Trade #"])
    entries = entries[entries["Trade #"].isin(common_trades)]
    exits = exits[exits["Trade #"].isin(common_trades)]
    
    # 重新設置索引
    entries = entries.set_index("Trade #")
    exits = exits.set_index("Trade #")

    # 合併 Entry → Exit
    pairs = entries[["Signal"]].rename(columns={"Signal": "Entry_Signal"})
    pairs["Exit_Signal"] = exits["Signal"]

    # 計算分布
    link_counts = pairs.groupby(["Entry_Signal", "Exit_Signal"]).size().reset_index(name="Count")

    # 內圈 (Entry)
    inner_labels = link_counts["Entry_Signal"].unique()
    inner_values = [link_counts.loc[link_counts["Entry_Signal"] == sig, "Count"].sum() for sig in inner_labels]

    # 外圈 (Exit) - 用唯一 key，顯示時只用 Exit 名稱
    outer_keys = [f"{row.Entry_Signal}|{row.Exit_Signal}" for row in link_counts.itertuples()]
    outer_labels = [row.Exit_Signal for row in link_counts.itertuples()]
    outer_values = link_counts["Count"].tolist()
    outer_parents = link_counts["Entry_Signal"].tolist()

    # 畫 nested pie chart (Sunburst)
    fig = go.Figure(go.Sunburst(
        ids=list(inner_labels) + outer_keys,       # 唯一 ID
        labels=list(inner_labels) + outer_labels,  # 顯示用的文字
        parents=[""] * len(inner_labels) + outer_parents,
        values=list(inner_values) + outer_values,
        branchvalues="total",
        insidetextorientation="radial",
        textinfo="label+percent entry",  # 顯示標籤和百分比
        textfont=dict(size=14),          # 調整字體大小以提高可讀性
        hoverinfo="label+percent entry+value",  # 修正為有效值
        marker=dict(
            line=dict(width=1, color="white")  # 區塊間的邊框
        )
    ))

    fig.update_layout(
        title_text="Entry vs Exit Signal Distribution (Nested Pie)",
        margin=dict(t=50, l=25, r=25, b=25),
        showlegend=False  # 關閉圖例以避免重複顯示標籤
    )
    return fig


def sankey_diagram(df: pd.DataFrame) -> go.Figure:
    # 將 Entry / Exit 拆開
    entries = df[df["Type"].str.contains("Entry")].set_index("Trade #")
    exits = df[df["Type"].str.contains("Exit")].set_index("Trade #")

    # 合併成一筆交易的 Entry -> Exit
    pairs = entries[["Signal"]].rename(columns={"Signal":"Entry_Signal"})
    pairs["Exit_Signal"] = exits["Signal"]

    # 計算分布
    link_counts = pairs.groupby(["Entry_Signal","Exit_Signal"]).size().reset_index(name="Count")

    # 建立節點列表
    all_signals = list(pd.concat([link_counts["Entry_Signal"], link_counts["Exit_Signal"]]).unique())
    signal_to_id = {signal:i for i,signal in enumerate(all_signals)}

    # Sankey 需要 source, target, value
    sources = link_counts["Entry_Signal"].map(signal_to_id)
    targets = link_counts["Exit_Signal"].map(signal_to_id)
    values = link_counts["Count"]

    # 畫 Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_signals
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text="Entry vs Exit Signal Flow", 
        # font_size=12,
        font=dict(size=12, color="black", family="Arial Black"),
        height=600,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    return fig


def create_stacked_bar_chart(df: pd.DataFrame):
    """創建 Stacked Bar Chart：橫軸是 Net P&L %，縱軸是交易數量，按 Entry Signal 分類"""
    entries = df[df["Type"].str.contains("Entry")].copy()
    
    # 創建 Net P&L % 的區間
    entries['Net_PL_Range'] = pd.cut(
        entries['Net P&L %'], 
        bins=[-float('inf'), -10, -5, 0, 5, 10, 15, 20, float('inf')],
        labels=['< -10%', '-10% to -5%', '-5% to 0%', '0% to 5%', '5% to 10%', '10% to 15%', '15% to 20%', '> 20%']
    )
    
    # 計算每個 Entry Signal 在各區間的數量
    stacked_data = entries.groupby(['Signal', 'Net_PL_Range']).size().reset_index(name='Count')
    
    # pivot table
    pivot_data = stacked_data.pivot(index='Net_PL_Range', columns='Signal', values='Count').fillna(0)
    
    # **倒序橫軸**
    pivot_data = pivot_data.iloc[::-1]
    
    # 使用 plotly 創建 stacked bar chart
    fig = go.Figure()
    
    for signal in pivot_data.columns:
        fig.add_trace(go.Bar(
            name=signal,
            x=pivot_data.index,
            y=pivot_data[signal],
            hovertemplate=f'<b>{signal}</b><br>區間: %{{x}}<br>交易數量: %{{y}}<extra></extra>',
            text=pivot_data[signal],        # 顯示每個 bar 上的數量
            textposition='auto'             # 自動放在 bar 上方
        ))
    
    # 更新布局
    fig.update_layout(
        title='交易數量分布 (按 Net P&L % 區間和 Entry Signal)',
        xaxis_title='Net P&L % 區間',
        yaxis_title='交易數量',
        barmode='stack',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # 旋轉 x 軸標籤以避免重疊
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_entry_exit_profit_analysis(df: pd.DataFrame):
    """分析不同進場策略到不同出場策略的 Net Profit 表現"""
    # 取 Entry / Exit，並按 Trade # 排序確保正確配對
    entries = df[df["Type"].str.contains("Entry")].copy()
    exits = df[df["Type"].str.contains("Exit")].copy()
    
    # 按 Trade # 排序
    entries = entries.sort_values("Trade #")
    exits = exits.sort_values("Trade #")
    
    # 確保每個 Trade # 都有對應的 Entry 和 Exit
    common_trades = set(entries["Trade #"]) & set(exits["Trade #"])
    entries = entries[entries["Trade #"].isin(common_trades)]
    exits = exits[exits["Trade #"].isin(common_trades)]
    
    # 重新設置索引
    entries = entries.set_index("Trade #")
    exits = exits.set_index("Trade #")

    # 合併 Entry → Exit
    pairs = entries[["Signal"]].rename(columns={"Signal": "Entry_Signal"})
    pairs["Exit_Signal"] = exits["Signal"]
    pairs["Net_PL_USDT"] = entries["Net P&L USDT"]
    pairs["Net_PL_Percent"] = entries["Net P&L %"]
    pairs["Trade_Count"] = 1  # 用於計算交易數量

    # 計算統計數據
    stats = pairs.groupby(["Entry_Signal", "Exit_Signal"]).agg({
        "Net_PL_USDT": ["sum", "mean", "count"],
        "Net_PL_Percent": ["mean", "std"]
    }).round(2)
    
    # 扁平化多層索引
    stats.columns = ["Total_PL_USDT", "Avg_PL_USDT", "Trade_Count", "Avg_PL_Percent", "Std_PL_Percent"]
    stats = stats.reset_index()
    
    return pairs, stats

def create_profit_heatmap(df: pd.DataFrame):
    """創建進場-出場策略的獲利熱力圖"""
    pairs, stats = create_entry_exit_profit_analysis(df)
    
    # 創建 pivot table 用於熱力圖
    heatmap_data = stats.pivot(index="Entry_Signal", columns="Exit_Signal", values="Avg_PL_Percent")
    
    # 使用 plotly 創建熱力圖
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',  # 紅-黃-綠色彩，綠色表示獲利
        zmid=0,  # 以0為中心
        text=heatmap_data.values,
        texttemplate="%{text:.1f}%",
        textfont={"size": 12},
        hovertemplate='<b>%{y} → %{x}</b><br>平均獲利: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="進場-出場策略獲利熱力圖 (平均 Net P&L %)",
        xaxis_title="出場策略",
        yaxis_title="進場策略",
        height=500,
        font=dict(size=12)
    )
    
    return fig

def create_profit_table(df: pd.DataFrame):
    """創建進場-出場策略的詳細統計表格"""
    pairs, stats = create_entry_exit_profit_analysis(df)
    
    # 重新排列和格式化表格
    display_stats = stats.copy()
    display_stats = display_stats.sort_values(["Entry_Signal", "Exit_Signal"])
    
    # 格式化數值
    display_stats["Total_PL_USDT"] = display_stats["Total_PL_USDT"].apply(lambda x: f"{x:,.2f}")
    display_stats["Avg_PL_USDT"] = display_stats["Avg_PL_USDT"].apply(lambda x: f"{x:,.2f}")
    display_stats["Avg_PL_Percent"] = display_stats["Avg_PL_Percent"].apply(lambda x: f"{x:.2f}%")
    display_stats["Std_PL_Percent"] = display_stats["Std_PL_Percent"].apply(lambda x: f"{x:.2f}%")
    
    # 重新命名欄位
    display_stats.columns = [
        "進場策略", "出場策略", "總獲利 (USDT)", "平均獲利 (USDT)", 
        "交易次數", "平均獲利 (%)", "獲利標準差 (%)"
    ]
    
    return display_stats

@st.fragment
def create_profit_summary_chart(df: pd.DataFrame):
    """創建進場-出場策略的獲利摘要圖表"""
    pairs, stats = create_entry_exit_profit_analysis(df)
    entry_signals = stats["Entry_Signal"].unique().tolist()
    selected_entry_signal = st.selectbox("Select Entry Signal", ["All"] + entry_signals)

    if selected_entry_signal != "All":
        stats_filtered = stats[stats["Entry_Signal"] == selected_entry_signal].copy()
    else:
        stats_filtered = stats

    # 計算每個進場策略的總獲利
    entry_summary = stats.groupby("Entry_Signal").agg({
        "Total_PL_USDT": "sum",
        "Trade_Count": "sum",
        "Avg_PL_Percent": "mean"
    }).reset_index()
    
    # 計算每個出場策略的總獲利
    exit_summary = stats_filtered.groupby("Exit_Signal").agg({
        "Total_PL_USDT": "sum",
        "Trade_Count": "sum",
        "Avg_PL_Percent": "mean"
    }).reset_index()
    
    # 創建子圖
    from plotly.subplots import make_subplots
    

    cols = st.columns(2)
    with cols[0]:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("進場策略總獲利", "進場策略平均獲利%"),
            specs=[[{"type": "bar"}],
                [{"type": "bar"}]]
        )
        
        # 進場策略總獲利
        fig.add_trace(
            go.Bar(x=entry_summary["Entry_Signal"], y=entry_summary["Total_PL_USDT"], 
                name="進場總獲利", marker_color="lightblue"),
            row=1, col=1
        )
        
        # 進場策略平均獲利%
        fig.add_trace(
            go.Bar(x=entry_summary["Entry_Signal"], y=entry_summary["Avg_PL_Percent"], 
                name="進場平均獲利%", marker_color="orange"),
            row=2, col=1
        )
        fig.update_layout(
            title="進場策略獲利摘要",
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("出場策略總獲利", "出場策略平均獲利%"),
            specs=[[{"type": "bar"}],
                [{"type": "bar"}]]
        )
        
        # 出場策略總獲利
        fig.add_trace(
            go.Bar(x=exit_summary["Exit_Signal"], y=exit_summary["Total_PL_USDT"], 
                name="出場總獲利", marker_color="lightgreen"),
            row=1, col=1
        )
        
        # 出場策略平均獲利%
        fig.add_trace(
            go.Bar(x=exit_summary["Exit_Signal"], y=exit_summary["Avg_PL_Percent"], 
                name="出場平均獲利%", marker_color="red"),
            row=2, col=1
        )
        
        if selected_entry_signal != "All":
            title_text = f"出場策略獲利摘要 (進場策略: {selected_entry_signal})"
        else:
            title_text = "出場策略獲利摘要"
        
        fig.update_layout(
            title=title_text,
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

@st.fragment
def create_profit_loss_analysis(df: pd.DataFrame):
    """分析不同進場策略的虧損和獲利情況，使用柏拉圖和直方圖"""
    # 只取 Entry 記錄
    entries = df[df["Type"].str.contains("Entry")].copy()
    
    # 獲取所有進場策略
    entry_signals = entries["Signal"].unique().tolist()
    selected_signal = st.selectbox("選擇進場策略", entry_signals, key="profit_loss_signal")
    
    # 篩選選定策略的數據
    signal_data = entries[entries["Signal"] == selected_signal].copy()
    
    # 分離獲利和虧損交易
    profitable_trades = signal_data[signal_data["Net P&L %"] > 0].copy()
    loss_trades = signal_data[signal_data["Net P&L %"] <= 0].copy()
    
    # 創建兩個欄位
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📈 {selected_signal} - 獲利分析")
        
        if len(profitable_trades) > 0:
            # 獲利柏拉圖
            st.write("**獲利柏拉圖 (80/20法則)**")
            create_pareto_chart(profitable_trades, "Net P&L %", "獲利交易", "獲利%")
            
            # 獲利直方圖
            st.write("**獲利分布直方圖**")
            create_histogram_chart(profitable_trades, "Net P&L %", "獲利%", "交易次數", "獲利交易分布")
        else:
            st.info("該策略沒有獲利交易")
    
    with col2:
        st.subheader(f"📉 {selected_signal} - 虧損分析")
        
        if len(loss_trades) > 0:
            # 虧損柏拉圖
            st.write("**虧損柏拉圖 (80/20法則)**")
            create_pareto_chart(loss_trades, "Net P&L %", "虧損交易", "虧損%")
            
            # 虧損直方圖
            st.write("**虧損分布直方圖**")
            create_histogram_chart(loss_trades, "Net P&L %", "虧損%", "交易次數", "虧損交易分布")
        else:
            st.info("該策略沒有虧損交易")
    
    # 顯示統計摘要
    st.subheader("📊 統計摘要")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總交易數", len(signal_data))
    with col2:
        st.metric("獲利交易數", len(profitable_trades))
    with col3:
        st.metric("虧損交易數", len(loss_trades))
    with col4:
        win_rate = len(profitable_trades) / len(signal_data) * 100 if len(signal_data) > 0 else 0
        st.metric("勝率", f"{win_rate:.1f}%")

def create_pareto_chart(data: pd.DataFrame, value_col: str, title_prefix: str, x_label: str):
    """創建柏拉圖圖表，分析80/20法則"""
    # 創建獲利/虧損區間
    if title_prefix == "獲利交易":
        # 獲利交易：從高到低排序
        bins = [0, 5, 10, 15, 20, 25, 30, float('inf')]
        labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30%+']
    else:
        # 虧損交易：從低到高排序（虧損程度）
        bins = [float('-inf'), -30, -25, -20, -15, -10, -5, 0]
        labels = ['-30%以下', '-25%到-30%', '-20%到-25%', '-15%到-20%', '-10%到-15%', '-5%到-10%', '0%到-5%']
    
    # 將數據分配到區間
    data['range'] = pd.cut(data[value_col], bins=bins, labels=labels, include_lowest=True)
    
    # 計算每個區間的交易次數
    range_counts = data['range'].value_counts()
    
    # 按交易次數從多到少排序（真正的柏拉圖排序）
    range_counts = range_counts.sort_values(ascending=False)
    
    # 計算累積百分比（基於交易次數的累積）
    cumulative_percent = (range_counts.cumsum() / range_counts.sum() * 100)
    
    # 創建雙軸圖表
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # 添加柱狀圖（交易次數）
    fig.add_trace(
        go.Bar(
            x=range_counts.index,
            y=range_counts.values,
            name='交易次數',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate=f'<b>%{{x}}</b><br>交易次數: %{{y}}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # 添加累積百分比線
    fig.add_trace(
        go.Scatter(
            x=range_counts.index,
            y=cumulative_percent.values,
            mode='lines+markers',
            name='累積百分比',
            line=dict(color='orange', width=3),
            marker=dict(size=6, color='red'),
            hovertemplate=f'<b>%{{x}}</b><br>累積百分比: %{{y:.1f}}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    # 添加80%參考線
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="80%線", annotation_position="top right",
                  secondary_y=True)
    
    # 找出80%的關鍵點
    if len(cumulative_percent) > 0:
        # 找到貢獻80%交易次數的區間
        key_ranges = cumulative_percent[cumulative_percent <= 80]
        if len(key_ranges) > 0:
            key_count = len(key_ranges)
            total_ranges = len(range_counts)
            key_percent = key_count / total_ranges * 100
            st.info(f"**80/20法則分析**: {key_count}個區間 ({key_percent:.1f}%) 包含了80%的交易次數")
    
    # 設置軸標籤
    fig.update_xaxes(title_text=f"{title_prefix}區間 (%)")
    fig.update_yaxes(title_text="交易次數", secondary_y=False)
    fig.update_yaxes(title_text="累積百分比 (%)", secondary_y=True)
    
    # 旋轉X軸標籤以避免重疊
    fig.update_xaxes(tickangle=45)
    
    # 更新布局
    fig.update_layout(
        title=f"{title_prefix} - 柏拉圖分析",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_histogram_chart(data: pd.DataFrame, value_col: str, x_label: str, y_label: str, title: str):
    """創建直方圖"""
    fig = go.Figure()
    
    # 創建直方圖
    fig.add_trace(go.Histogram(
        x=data[value_col],
        nbinsx=20,  # 設定bin數量
        name=title,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # 添加統計線
    mean_val = data[value_col].mean()
    median_val = data[value_col].median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"平均: {mean_val:.2f}%", annotation_position="top")
    fig.add_vline(x=median_val, line_dash="dot", line_color="green", 
                  annotation_text=f"中位數: {median_val:.2f}%", annotation_position="bottom")
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    


def analyze_entry_performance(df: pd.DataFrame):
    st.subheader("📈 進出場訊號分布")
    fig_sankey = sankey_diagram(df)
    st.plotly_chart(fig_sankey, use_container_width=True)

    fig_nested_pie = nested_pie_chart(df)
    st.plotly_chart(fig_nested_pie, use_container_width=True)
    
    # 新增 Stacked Bar Chart
    st.subheader("📋 交易數量分布 (按 Net P&L % 區間)")
    stacked_fig = create_stacked_bar_chart(df)
    st.plotly_chart(stacked_fig, use_container_width=True)
    
    # 新增進場-出場策略獲利分析
    st.subheader("💰 進場-出場策略獲利分析")
    tabs = st.tabs(["獲利熱力圖", "詳細統計表格"])
    with tabs[0]:
        # 熱力圖
        heatmap_fig = create_profit_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    with tabs[1]:
        # 詳細統計表格
        profit_table = create_profit_table(df)
        st.dataframe(profit_table, use_container_width=True)
    
    # 獲利摘要圖表
    st.markdown("**獲利摘要圖表**")
    create_profit_summary_chart(df)

    # # 獲利摘要圖表
    # st.markdown("**虧損/獲利分析**")
    # create_profit_loss_analysis(df)

    # 只取 Entry
    entries = df[df["Type"].str.contains("Entry")].copy()

    # 選擇要分析的欄位
    metrics = ["Net P&L %", "Drawdown %"]

    # ========== 1. 箱型圖 / 小提琴圖 ==========
    st.subheader("📦 P&L 分布 (依進場策略)")

    fig_violin = px.violin(entries, x="Signal", y="Net P&L %", color="Signal",
                           box=True, points="all",
                           title="各進場策略的獲利分布 (Violin Plot)")
    st.plotly_chart(fig_violin, use_container_width=True)

    # ========== 2. 平均績效 Bar Chart ==========
    st.subheader("📊 平均獲利 & 平均回撤 (依進場策略)")
    st.markdown("""
- Net P&L → 交易結束後的真正盈虧 → 結果
- Drawdown → 交易過程中可能出現的最大帳面虧損 → 過程中風險大小""")
    avg_stats = entries.groupby("Signal")[metrics].mean().reset_index()

    fig_bar = px.bar(avg_stats, x="Signal", y=["Net P&L %", "Drawdown %"],
                     barmode="group", title="各進場策略平均 P&L 與 Drawdown")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ========== 4. 總體統計表 ==========
    st.subheader("📑 各進場策略績效統計")
    total_stats = entries.groupby("Signal")[metrics].agg(["mean", "median", "max", "min", "count"])
    st.dataframe(total_stats)

def main() -> None:
    st.set_page_config(
        page_title="Trades",
        page_icon="📊",
        layout="wide"
    )
    st.title("List of Trades Analysis")

    base_dir = Path(__file__).parent.parent
    csv_dir = os.path.join(base_dir, "ListofTrades")
    default_csv = find_default_csv_file(base_dir)

    csv_pattern = os.path.join(csv_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    if not csv_files:
        st.error(f"No CSV files found in {csv_dir}. Please add results.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        labels = [os.path.basename(p) for p in csv_files]
        selected_idx = 0
        if default_csv in csv_files:
            selected_idx = csv_files.index(default_csv)
        selected_label = st.selectbox("Select CSV in ListofTrades/", labels, index=selected_idx)
        selected_path = csv_files[labels.index(selected_label)]

        

    csv_source = selected_path
    if isinstance(csv_source, str) and not os.path.exists(csv_source):
        st.error("The specified CSV path does not exist.")
        st.stop()

    df = load_results(csv_source)
    # st.write(df)

    analyze_entry_performance(df)

if __name__ == "__main__":
    main()