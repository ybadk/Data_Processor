"""
Visualization utilities for the Data Wrangling Application
Handles dashboard creation and data plotting
"""
# Machine Learning Modules
from datasets import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Handles all visualization and dashboard creation"""

    def __init__(self):
        self.color_palette = ['#0000CD', '#000088',
                              '#00FFFF', '#000080', '#ADD8E6', '#87CEFA']
        self.theme = {
            'background_color': '#000000',
            'text_color': '#FFFFFF',
            'grid_color': '#333333'
        }

    def create_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a comprehensive dashboard for the dataset"""
        try:
            dashboard_components = {}

            # Basic statistics
            dashboard_components['basic_stats'] = self._create_basic_stats(df)

            # Data quality overview
            dashboard_components['data_quality'] = self._create_data_quality_overview(
                df)

            # Numeric columns analysis
            numeric_cols = df.select_dtypes(
                include=[np.number]).columns.tolist()
            if numeric_cols:
                dashboard_components['numeric_analysis'] = self._create_numeric_analysis(
                    df, numeric_cols)

            # Categorical columns analysis
            categorical_cols = df.select_dtypes(
                include=['object']).columns.tolist()
            if categorical_cols:
                dashboard_components['categorical_analysis'] = self._create_categorical_analysis(
                    df, categorical_cols)

            # Correlation analysis
            if len(numeric_cols) > 1:
                dashboard_components['correlation'] = self._create_correlation_analysis(
                    df, numeric_cols)

            # Time series analysis (if date columns exist)
            date_cols = self._detect_date_columns(df)
            if date_cols:
                dashboard_components['time_series'] = self._create_time_series_analysis(
                    df, date_cols)

            return dashboard_components

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            st.error(f"Dashboard creation failed: {str(e)}")
            return {}

    def _create_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create basic statistics overview"""
        try:
            stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'data_types': df.dtypes.value_counts().to_dict()
            }

            return stats

        except Exception as e:
            logger.error(f"Error creating basic stats: {str(e)}")
            return {}

    def _create_data_quality_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create data quality visualization"""
        try:
            # Calculate missing values per column
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100

            # Create bar chart for missing values
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=missing_data.index,
                y=missing_percentage.values,
                name='Missing Data %',
                marker_color=self.color_palette[0],
                text=[f'{val:.1f}%' for val in missing_percentage.values],
                textposition='auto'
            ))

            fig.update_layout(
                title='Data Quality Overview - Missing Values by Column',
                xaxis_title='Columns',
                yaxis_title='Missing Data Percentage',
                template='plotly_dark',
                height=400
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating data quality overview: {str(e)}")
            return go.Figure()

    def _create_numeric_analysis(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, go.Figure]:
        """Create numeric columns analysis"""
        try:
            figures = {}

            # Distribution plots
            if len(numeric_cols) <= 4:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=numeric_cols[:4],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )

                for i, col in enumerate(numeric_cols[:4]):
                    row = (i // 2) + 1
                    col_pos = (i % 2) + 1

                    fig.add_trace(
                        go.Histogram(
                            x=df[col].dropna(),
                            name=col,
                            marker_color=self.color_palette[i % len(
                                self.color_palette)],
                            showlegend=True
                        ),
                        row=row, col=col_pos
                    )

                fig.update_layout(
                    title='Distribution of Numeric Columns',
                    template='plotly_dark',
                    height=600
                )

                figures['distributions'] = fig

            # Box plots for outlier detection
            fig_box = go.Figure()

            for i, col in enumerate(numeric_cols[:6]):  # Limit to 6 columns
                fig_box.add_trace(go.Box(
                    y=df[col].dropna(),
                    name=col,
                    marker_color=self.color_palette[i % len(
                        self.color_palette)]
                ))

            fig_box.update_layout(
                title='Box Plots - Outlier Detection',
                template='plotly_dark',
                height=400
            )

            figures['box_plots'] = fig_box

            return figures

        except Exception as e:
            logger.error(f"Error creating numeric analysis: {str(e)}")
            return {}

    def _create_categorical_analysis(self, df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create categorical columns analysis"""
        try:
            figures = {}

            # Value counts for categorical columns
            for col in categorical_cols[:3]:  # Limit to 3 columns
                value_counts = df[col].value_counts().head(10)

                fig = go.Figure(data=[
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        marker_color=self.color_palette[0],
                        text=value_counts.values,
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    title=f'Top 10 Values in {col}',
                    xaxis_title=col,
                    yaxis_title='Count',
                    template='plotly_dark',
                    height=400
                )

                figures[f'{col}_counts'] = fig

            # Pie chart for first categorical column
            if categorical_cols:
                col = categorical_cols[0]
                value_counts = df[col].value_counts().head(8)

                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=value_counts.index,
                        values=value_counts.values,
                        marker_colors=self.color_palette
                    )
                ])

                fig_pie.update_layout(
                    title=f'Distribution of {col}',
                    template='plotly_dark',
                    height=400
                )

                figures['pie_chart'] = fig_pie

            return figures

        except Exception as e:
            logger.error(f"Error creating categorical analysis: {str(e)}")
            return {}

    def _create_correlation_analysis(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        try:
            correlation_matrix = df[numeric_cols].corr()

            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))

            fig.update_layout(
                title='Correlation Matrix',
                template='plotly_dark',
                height=500,
                width=500
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating correlation analysis: {str(e)}")
            return go.Figure()

    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect potential date columns"""
        date_cols = []

        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as date
                try:
                    pd.to_datetime(df[col].dropna().head(100), errors='raise')
                    date_cols.append(col)
                except:
                    pass

        return date_cols

    def _create_time_series_analysis(self, df: pd.DataFrame, date_cols: List[str]) -> Dict[str, go.Figure]:
        """Create time series analysis"""
        try:
            figures = {}

            for date_col in date_cols[:2]:  # Limit to 2 date columns
                # Convert to datetime if not already
                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(
                        df[date_col], errors='coerce')

                # Create time series plot
                df_sorted = df.sort_values(date_col)

                fig = go.Figure()

                # Count of records over time
                time_counts = df_sorted.groupby(
                    df_sorted[date_col].dt.date).size()

                fig.add_trace(go.Scatter(
                    x=time_counts.index,
                    y=time_counts.values,
                    mode='lines+markers',
                    name='Records Count',
                    line=dict(color=self.color_palette[0])
                ))

                fig.update_layout(
                    title=f'Time Series Analysis - {date_col}',
                    xaxis_title='Date',
                    yaxis_title='Count',
                    template='plotly_dark',
                    height=400
                )

                figures[f'{date_col}_timeseries'] = fig

            return figures

        except Exception as e:
            logger.error(f"Error creating time series analysis: {str(e)}")
            return {}

    def create_custom_plot(self, df: pd.DataFrame, plot_type: str,
                           x_col: str, y_col: str = None, **kwargs) -> go.Figure:
        """Create custom plots based on user selection"""
        try:
            fig = go.Figure()

            if plot_type == 'scatter':
                if y_col:
                    fig.add_trace(go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='markers',
                        marker=dict(color=self.color_palette[0])
                    ))
                    fig.update_layout(
                        title=f'Scatter Plot: {x_col} vs {y_col}',
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )

            elif plot_type == 'line':
                if y_col:
                    fig.add_trace(go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='lines',
                        line=dict(color=self.color_palette[0])
                    ))
                    fig.update_layout(
                        title=f'Line Plot: {x_col} vs {y_col}',
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )

            elif plot_type == 'bar':
                value_counts = df[x_col].value_counts().head(20)
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color=self.color_palette[0]
                ))
                fig.update_layout(
                    title=f'Bar Chart: {x_col}',
                    xaxis_title=x_col,
                    yaxis_title='Count'
                )

            elif plot_type == 'histogram':
                fig.add_trace(go.Histogram(
                    x=df[x_col],
                    marker_color=self.color_palette[0]
                ))
                fig.update_layout(
                    title=f'Histogram: {x_col}',
                    xaxis_title=x_col,
                    yaxis_title='Frequency'
                )

            fig.update_layout(
                template='plotly_dark',
                height=400
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating custom plot: {str(e)}")
            return go.Figure()

    def create_wordcloud(self, text_data: pd.Series) -> str:
        """Create word cloud from text data"""
        try:
            # Combine all text
            text = ' '.join(text_data.dropna().astype(str))

            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='black',
                colormap='viridis',
                max_words=100
            ).generate(text)

            # Convert to base64 for display
            img_buffer = io.BytesIO()
            wordcloud.to_image().save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()

            return img_str

        except Exception as e:
            logger.error(f"Error creating word cloud: {str(e)}")
            return ""

    def export_plots(self, figures: Dict[str, go.Figure], format: str = 'png') -> Dict[str, bytes]:
        """Export plots in specified format"""
        try:
            exported_plots = {}

            for name, fig in figures.items():
                if format == 'png':
                    img_bytes = fig.to_image(
                        format='png', width=800, height=600)
                elif format == 'html':
                    img_bytes = fig.to_html().encode('utf-8')
                else:
                    img_bytes = fig.to_image(
                        format='png', width=800, height=600)

                exported_plots[name] = img_bytes

            return exported_plots

        except Exception as e:
            logger.error(f"Error exporting plots: {str(e)}")
            return {}
