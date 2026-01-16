"""
DOGE多因子量化交易系统 - 稳定性优化版本
保留所有原始功能和数据，修复运行暂停问题
"""

import sys
import os
import json
import time
import threading
import queue
import argparse
import hmac
import hashlib
import urllib.parse
from functools import partial
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 数据获取库
import requests
import websocket
from bs4 import BeautifulSoup

# 机器学习库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

# 时间序列分析
from prophet import Prophet

# 深度学习库（PyTorch）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# 可视化库
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# GUI库
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import tkinter.font as tkFont

# ==================== 性能优化设置 ====================

# 设置日志
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('doge_trading.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 系统配置 ====================

class SystemConfig:
    """系统配置"""
    def __init__(self):
        # 数据源配置 - 保持原有数据量
        self.data_sources = {
            'price': {
                'symbol': 'DOGEUSDT',
                'intervals': ['1m', '5m', '15m', '1h', '4h', '1d'],
                'history_days': 365  # 保持1年历史数据
            },
            'social': {
                'twitter': True,
                'reddit': True,
                'news': True
            },
            'onchain': {
                'whale_tracking': True,
                'exchange_flows': True,
                'holder_distribution': True
            }
        }
        
        # 模型配置 - 保持所有模型
        self.model_config = {
            'online_learning': True,
            'retrain_interval': '1d',
            'ensemble_method': 'weighted',
            'models': ['xgb', 'lgb', 'rf', 'prophet', 'lstm', 'gbt']
        }
        
        # 交易配置 - 改进手续费计算
        self.trading_config = {
            'initial_balance': 10000.0,
            'max_position': 0.3,
            'risk_per_trade': 0.02,
            'stop_loss': 0.05,
            'take_profit': [0.08, 0.15],
            'commission_rate': 0.001,  # 手续费率
            'min_commission': 0.1,     # 最低手续费
            'commission_tiers': [      # 手续费分层
                {'threshold': 1000, 'rate': 0.002},
                {'threshold': 10000, 'rate': 0.0015},
                {'threshold': 50000, 'rate': 0.001},
                {'threshold': 100000, 'rate': 0.0005}
            ]
        }
        
        # 系统配置 - 优化性能
        self.system_config = {
            'data_refresh_interval': 60,
            'signal_check_interval': 300,
            'max_workers': 4,
            'cache_dir': 'cache',
            'model_dir': 'models',
            'enable_gc': True,          # 启用垃圾回收
            'gc_interval': 600,         # 垃圾回收间隔(秒)
            'max_queue_size': 1000,     # 最大队列大小
            'thread_timeout': 30,       # 线程超时时间(秒)
            'watchdog_interval': 300    # 看门狗检查间隔(秒)
        }
        
        # 创建必要目录
        os.makedirs(self.system_config['cache_dir'], exist_ok=True)
        os.makedirs(self.system_config['model_dir'], exist_ok=True)

# ==================== 稳定化的数据管理器 ====================

class StableDataManager:
    """稳定的数据管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.cache = {}
        self.last_update = {}
        
        # 数据队列 - 使用带超时的操作
        self.price_queue = queue.Queue(maxsize=config.system_config['max_queue_size'])
        self.social_queue = queue.Queue(maxsize=config.system_config['max_queue_size'] // 10)
        self.onchain_queue = queue.Queue(maxsize=config.system_config['max_queue_size'] // 10)
        
        # 数据存储
        self.historical_data = {
            'price': {},
            'social': pd.DataFrame(),  # 初始化为空的DataFrame
            'onchain': pd.DataFrame(),
            'derivatives': pd.DataFrame()
        }
        
        # 线程控制
        self.data_stream_thread = None
        self.stop_streaming = threading.Event()
        self.stream_lock = threading.RLock()
        
        # 性能监控
        self.update_count = 0
        self.last_gc_time = time.time()
        
        logger.info("稳定的数据管理器初始化完成")
    
    def fetch_all_historical_data(self):
        """获取所有历史数据"""
        logger.info("开始获取1年历史数据...")
        
        try:
            # 1. 获取价格数据
            self._fetch_price_history()
            
            # 2. 获取社交媒体数据
            self._fetch_social_sentiment_history()
            
            # 3. 获取链上数据
            self._fetch_onchain_data_history()
            
            # 4. 获取衍生品数据
            self._fetch_derivatives_data()
            
            # 执行垃圾回收
            self._perform_garbage_collection()
            
            logger.info("历史数据获取完成")
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            raise
    
    def _fetch_price_history(self):
        """获取价格历史数据"""
        try:
            # 生成1年分钟数据
            dates = pd.date_range(
                end=datetime.now(),
                periods=365*24*60,
                freq='1min'
            )
            
            np.random.seed(42)
            base_price = 0.08
            returns = np.random.normal(0, 0.02/365, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            # 添加季节性和趋势
            seasonal = 0.0005 * np.sin(2*np.pi*np.arange(len(dates))/(24*60))
            trend = np.linspace(0, 0.1, len(dates))
            prices *= (1 + seasonal + trend)
            
            # 生成OHLCV数据
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
                'high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
                'low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
                'close': prices,
                'volume': np.random.lognormal(12, 1.2, len(dates)) * 1000
            })
            
            # 计算技术指标
            df = self._calculate_technical_indicators(df)
            
            self.historical_data['price']['1m'] = df
            logger.info(f"价格数据获取完成: {len(df)} 条记录")
            
            # 生成其他时间框架数据
            self._resample_timeframes(df)
            
        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # 移动平均线
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
        
        # 指数移动平均线
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR (真实波动幅度)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 价格动量
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # 填充NaN值
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _resample_timeframes(self, df: pd.DataFrame):
        """重采样到不同时间框架"""
        df.set_index('timestamp', inplace=True)
        
        timeframes = {
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D'
        }
        
        for tf_key, tf_freq in timeframes.items():
            try:
                # 重采样
                resampled = df.resample(tf_freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # 重新计算指标
                resampled = self._calculate_technical_indicators(resampled.reset_index())
                self.historical_data['price'][tf_key] = resampled
                
                logger.info(f"{tf_key} 数据生成完成: {len(resampled)} 条记录")
                
            except Exception as e:
                logger.error(f"重采样 {tf_key} 失败: {e}")
        
        df.reset_index(inplace=True)
    
    def _fetch_social_sentiment_history(self):
        """获取社交媒体情绪历史"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=365,
                freq='1D'
            )
            
            np.random.seed(42)
            base_sentiment = 0.5
            
            trend = np.linspace(0, 0.3, len(dates))
            seasonal = 0.2 * np.sin(2*np.pi*np.arange(len(dates))/30)
            noise = np.random.normal(0, 0.1, len(dates))
            
            sentiment = base_sentiment + trend + seasonal + noise
            sentiment = np.clip(sentiment, 0, 1)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'twitter_sentiment': sentiment,
                'reddit_sentiment': sentiment * 0.8 + np.random.uniform(0, 0.2, len(dates)),
                'news_sentiment': sentiment * 0.7 + np.random.uniform(0, 0.3, len(dates)),
                'social_volume': np.random.lognormal(8, 1, len(dates)),
                'weighted_sentiment': sentiment * 0.4 + 
                                     sentiment * 0.8 * 0.3 + 
                                     sentiment * 0.7 * 0.3
            })
            
            self.historical_data['social'] = df
            logger.info(f"社交媒体数据生成完成: {len(df)} 条记录")
            
        except Exception as e:
            logger.error(f"获取社交媒体数据失败: {e}")
    
    def _fetch_onchain_data_history(self):
        """获取链上数据历史"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=365,
                freq='1D'
            )
            
            np.random.seed(42)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'active_addresses': np.random.randint(50000, 200000, len(dates)),
                'transaction_count': np.random.randint(10000, 50000, len(dates)),
                'exchange_inflow': np.random.uniform(-1000000, 1000000, len(dates)),
                'exchange_outflow': np.random.uniform(-1000000, 1000000, len(dates)),
                'whale_transactions': np.random.randint(10, 100, len(dates)),
                'network_growth': np.random.uniform(-0.05, 0.05, len(dates)),
                'hodl_waves_1d': np.random.uniform(0, 0.3, len(dates)),
                'hodl_waves_1w': np.random.uniform(0, 0.25, len(dates)),
                'hodl_waves_1m': np.random.uniform(0, 0.2, len(dates)),
                'hodl_waves_3m': np.random.uniform(0, 0.15, len(dates))
            })
            
            df['net_exchange_flow'] = df['exchange_inflow'] - df['exchange_outflow']
            
            self.historical_data['onchain'] = df
            logger.info(f"链上数据生成完成: {len(df)} 条记录")
            
        except Exception as e:
            logger.error(f"获取链上数据失败: {e}")
    
    def _fetch_derivatives_data(self):
        """获取衍生品数据"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=365,
                freq='1D'
            )
            
            np.random.seed(42)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'funding_rate': np.random.uniform(-0.01, 0.01, len(dates)),
                'open_interest': np.random.uniform(1000000, 5000000, len(dates)),
                'long_short_ratio': np.random.uniform(0.5, 1.5, len(dates)),
                'liquidations_long': np.random.uniform(0, 500000, len(dates)),
                'liquidations_short': np.random.uniform(0, 500000, len(dates))
            })
            
            self.historical_data['derivatives'] = df
            logger.info(f"衍生品数据生成完成: {len(df)} 条记录")
            
        except Exception as e:
            logger.error(f"获取衍生品数据失败: {e}")
    
    def start_data_streaming(self):
        """启动数据流"""
        if self.data_stream_thread and self.data_stream_thread.is_alive():
            logger.warning("数据流线程已在运行")
            return
        
        self.stop_streaming.clear()
        self.data_stream_thread = threading.Thread(
            target=self._stream_real_time_data,
            daemon=True,
            name="DataStreamThread"
        )
        self.data_stream_thread.start()
        logger.info("数据流线程已启动")
    
    def stop_data_streaming(self):
        """停止数据流"""
        self.stop_streaming.set()
        if self.data_stream_thread:
            self.data_stream_thread.join(timeout=5)
            logger.info("数据流线程已停止")
    
    def _stream_real_time_data(self):
        """流式获取实时数据"""
        logger.info("开始实时数据流")
        
        while not self.stop_streaming.is_set():
            try:
                with self.stream_lock:
                    # 模拟实时价格更新
                    current_time = datetime.now()
                    
                    # 获取最后价格
                    if '1m' in self.historical_data['price']:
                        last_price = self.historical_data['price']['1m']['close'].iloc[-1]
                    else:
                        last_price = 0.08
                    
                    new_return = np.random.normal(0, 0.0001)
                    new_price = last_price * (1 + new_return)
                    
                    # 创建新数据行
                    new_row = pd.DataFrame([{
                        'timestamp': current_time,
                        'open': new_price * (1 + np.random.uniform(-0.0005, 0.0005)),
                        'high': new_price * (1 + np.random.uniform(0, 0.001)),
                        'low': new_price * (1 - np.random.uniform(0, 0.001)),
                        'close': new_price,
                        'volume': np.random.lognormal(11, 1) * 1000
                    }])
                    
                    # 更新数据框 - 添加新行
                    if '1m' in self.historical_data['price']:
                        self.historical_data['price']['1m'] = pd.concat([
                            self.historical_data['price']['1m'],
                            new_row
                        ], ignore_index=True)
                    
                    # 放入队列（非阻塞）
                    try:
                        self.price_queue.put({
                            'timestamp': current_time,
                            'price': new_price,
                            'volume': new_row['volume'].iloc[0]
                        }, timeout=0.5)
                    except queue.Full:
                        # 队列满时尝试清理
                        try:
                            self.price_queue.get_nowait()
                        except queue.Empty:
                            pass
                
                self.update_count += 1
                
                # 定期执行垃圾回收
                current_time = time.time()
                if current_time - self.last_gc_time > 300:  # 每5分钟
                    self._perform_garbage_collection()
                    self.last_gc_time = current_time
                
                # 等待一段时间
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"实时数据流错误: {e}")
                # 出错后短暂等待
                time.sleep(5)
        
        logger.info("实时数据流已结束")
    
    def _perform_garbage_collection(self):
        """执行垃圾回收"""
        if self.config.system_config['enable_gc']:
            import gc
            gc.collect()
            logger.debug("垃圾回收已执行")
    
    def get_latest_data(self) -> Dict:
        """获取最新数据"""
        latest_data = {
            'timestamp': datetime.now(),
            'price': {},
            'social': {},
            'onchain': {},
            'derivatives': {}
        }
        
        try:
            with self.stream_lock:
                # 获取最新价格
                if '1m' in self.historical_data['price']:
                    price_df = self.historical_data['price']['1m']
                    if not price_df.empty:
                        latest_price = price_df.iloc[-1].to_dict()
                        latest_data['price'] = latest_price
                
                # 获取最新社交媒体情绪
                if isinstance(self.historical_data['social'], pd.DataFrame) and not self.historical_data['social'].empty:
                    social_df = self.historical_data['social']
                    latest_social = social_df.iloc[-1].to_dict()
                    latest_data['social'] = latest_social
                
                # 获取最新链上数据
                if isinstance(self.historical_data['onchain'], pd.DataFrame) and not self.historical_data['onchain'].empty:
                    onchain_df = self.historical_data['onchain']
                    latest_onchain = onchain_df.iloc[-1].to_dict()
                    latest_data['onchain'] = latest_onchain
                
                # 获取最新衍生品数据
                if isinstance(self.historical_data['derivatives'], pd.DataFrame) and not self.historical_data['derivatives'].empty:
                    derivatives_df = self.historical_data['derivatives']
                    latest_derivatives = derivatives_df.iloc[-1].to_dict()
                    latest_data['derivatives'] = latest_derivatives
            
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
        
        return latest_data

# ==================== 改进的特征工程师 ====================

class ImprovedFeatureEngineer:
    """改进的特征工程师"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.feature_cache = {}  # 特征缓存
        
    def create_multi_factor_features(self, 
                                    price_data: pd.DataFrame,
                                    social_data: pd.DataFrame,
                                    onchain_data: pd.DataFrame,
                                    derivatives_data: pd.DataFrame) -> pd.DataFrame:
        """创建多因子特征"""
        
        logger.info("开始创建多因子特征...")
        
        # 使用缓存提高性能
        cache_key = self._generate_cache_key(price_data, social_data, onchain_data, derivatives_data)
        if cache_key in self.feature_cache:
            logger.info("使用缓存的特征数据")
            return self.feature_cache[cache_key]
        
        # 对齐数据时间
        aligned_df = self._align_data_sources(
            price_data, social_data, onchain_data, derivatives_data
        )
        
        if aligned_df.empty:
            logger.warning("对齐后的数据为空")
            return pd.DataFrame()
        
        # 创建特征数据框
        features_df = pd.DataFrame(index=aligned_df.index)
        
        # 1. 价格和技术特征
        features_df = self._create_price_features(features_df, aligned_df)
        
        # 2. 社交媒体情绪特征
        features_df = self._create_social_features(features_df, aligned_df)
        
        # 3. 链上数据特征
        features_df = self._create_onchain_features(features_df, aligned_df)
        
        # 4. 衍生品市场特征
        features_df = self._create_derivatives_features(features_df, aligned_df)
        
        # 5. 交互特征
        features_df = self._create_interaction_features(features_df)
        
        # 6. 时间特征
        features_df = self._create_time_features(features_df)
        
        # 7. 市场状态特征
        features_df = self._create_market_regime_features(features_df, aligned_df)
        
        # 处理缺失值
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # 记录特征列
        self.feature_columns = features_df.columns.tolist()
        
        # 缓存结果
        self.feature_cache[cache_key] = features_df
        
        logger.info(f"特征创建完成，共 {len(self.feature_columns)} 个特征")
        
        return features_df
    
    def _generate_cache_key(self, *dataframes):
        """生成缓存键"""
        key_parts = []
        for df in dataframes:
            if df is not None and not df.empty:
                key_parts.append(f"shape:{df.shape}_last:{df.index[-1] if hasattr(df, 'index') else len(df)}")
        return "_".join(key_parts) if key_parts else "empty"
    
    def _align_data_sources(self, price_df, social_df, onchain_df, derivatives_df):
        """对齐不同数据源的时间"""
        
        try:
            # 将所有数据框转换为副本
            price_df = price_df.copy() if price_df is not None else pd.DataFrame()
            social_df = social_df.copy() if social_df is not None else pd.DataFrame()
            onchain_df = onchain_df.copy() if onchain_df is not None else pd.DataFrame()
            derivatives_df = derivatives_df.copy() if derivatives_df is not None else pd.DataFrame()
            
            # 确保price_df有timestamp列
            if 'timestamp' not in price_df.columns and not price_df.empty:
                if price_df.index.name == 'timestamp':
                    price_df = price_df.reset_index()
            
            # 设置timestamp为索引
            dfs_to_process = []
            for df, name in [(price_df, 'price'), (social_df, 'social'), 
                           (onchain_df, 'onchain'), (derivatives_df, 'derivatives')]:
                if not df.empty and 'timestamp' in df.columns:
                    df_copy = df.copy()
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
                    df_copy.set_index('timestamp', inplace=True)
                    dfs_to_process.append((df_copy, name))
            
            if not dfs_to_process:
                return pd.DataFrame()
            
            # 以价格数据为基础，对齐其他数据
            base_df = dfs_to_process[0][0] if dfs_to_process else pd.DataFrame()
            
            # 合并所有数据
            aligned_df = base_df[['close', 'volume']].copy() if 'close' in base_df.columns else base_df.copy()
            
            for df, name in dfs_to_process[1:]:
                # 只合并数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    aligned_df = aligned_df.join(df[numeric_cols], how='left', rsuffix=f'_{name}')
            
            # 填充缺失值
            aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
            
            return aligned_df
            
        except Exception as e:
            logger.error(f"数据对齐失败: {e}")
            # 返回空数据框避免崩溃
            return pd.DataFrame()
    
    def _create_price_features(self, features_df, aligned_df):
        """创建价格特征"""
        
        if 'close' in aligned_df.columns:
            # 收益率特征
            for period in [1, 3, 5, 10, 20]:
                features_df[f'return_{period}d'] = aligned_df['close'].pct_change(period)
            
            # 波动率特征
            features_df['volatility_20d'] = aligned_df['close'].pct_change().rolling(20).std()
            features_df['volatility_50d'] = aligned_df['close'].pct_change().rolling(50).std()
            
            # 价格位置
            features_df['price_rank_20d'] = aligned_df['close'].rolling(20).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
            )
            
            # 成交量特征
            if 'volume' in aligned_df.columns:
                features_df['volume_ratio_20d'] = aligned_df['volume'] / aligned_df['volume'].rolling(20).mean()
                features_df['volume_price_corr_20d'] = aligned_df['close'].rolling(20).corr(aligned_df['volume'])
        
        return features_df
    
    def _create_social_features(self, features_df, aligned_df):
        """创建社交媒体特征"""
        
        social_metrics = ['twitter_sentiment', 'reddit_sentiment', 'news_sentiment',
                         'social_volume', 'weighted_sentiment']
        
        for metric in social_metrics:
            if metric in aligned_df.columns:
                # 原始值
                features_df[f'{metric}'] = aligned_df[metric]
                
                # 变化率
                features_df[f'{metric}_change'] = aligned_df[metric].pct_change()
                
                # 移动平均
                features_df[f'{metric}_ma7'] = aligned_df[metric].rolling(7).mean()
                features_df[f'{metric}_ma30'] = aligned_df[metric].rolling(30).mean()
        
        # 社交媒体综合指标
        if all(m in aligned_df.columns for m in ['twitter_sentiment', 'reddit_sentiment', 'news_sentiment']):
            features_df['social_sentiment_composite'] = (
                aligned_df['twitter_sentiment'] * 0.4 +
                aligned_df['reddit_sentiment'] * 0.3 +
                aligned_df['news_sentiment'] * 0.3
            )
        
        return features_df
    
    def _create_onchain_features(self, features_df, aligned_df):
        """创建链上特征"""
        
        onchain_metrics = [
            'active_addresses', 'transaction_count', 'net_exchange_flow',
            'whale_transactions', 'network_growth'
        ]
        
        for metric in onchain_metrics:
            if metric in aligned_df.columns:
                # 原始值
                features_df[f'{metric}'] = aligned_df[metric]
                
                # 变化率
                features_df[f'{metric}_change'] = aligned_df[metric].pct_change()
                
                # 标准化值
                mean_val = aligned_df[metric].rolling(30).mean()
                std_val = aligned_df[metric].rolling(30).std()
                features_df[f'{metric}_zscore'] = (
                    (aligned_df[metric] - mean_val) / std_val
                ).replace([np.inf, -np.inf], 0)
        
        # 链上综合指标
        if 'net_exchange_flow' in aligned_df.columns:
            features_df['exchange_flow_signal'] = np.where(
                aligned_df['net_exchange_flow'] > 0, 1, -1
            )
        
        if 'whale_transactions' in aligned_df.columns:
            features_df['whale_activity'] = (
                aligned_df['whale_transactions'] > 
                aligned_df['whale_transactions'].rolling(30).mean()
            ).astype(int)
        
        return features_df
    
    def _create_derivatives_features(self, features_df, aligned_df):
        """创建衍生品特征"""
        
        if 'funding_rate' in aligned_df.columns:
            features_df['funding_rate'] = aligned_df['funding_rate']
            features_df['funding_rate_abs'] = np.abs(aligned_df['funding_rate'])
            features_df['funding_rate_signal'] = np.where(
                aligned_df['funding_rate'] > 0.001, -1,
                np.where(aligned_df['funding_rate'] < -0.001, 1, 0)
            )
        
        if 'long_short_ratio' in aligned_df.columns:
            features_df['long_short_ratio'] = aligned_df['long_short_ratio']
            features_df['lsr_deviation'] = (
                aligned_df['long_short_ratio'] - 
                aligned_df['long_short_ratio'].rolling(30).mean()
            ) / aligned_df['long_short_ratio'].rolling(30).std()
        
        if all(m in aligned_df.columns for m in ['liquidations_long', 'liquidations_short']):
            features_df['liquidation_ratio'] = (
                aligned_df['liquidations_long'] / 
                (aligned_df['liquidations_short'] + 1e-10)
            )
            features_df['liquidation_imbalance'] = (
                aligned_df['liquidations_long'] - 
                aligned_df['liquidations_short']
            ) / (aligned_df['liquidations_long'] + aligned_df['liquidations_short'] + 1e-10)
        
        return features_df
    
    def _create_interaction_features(self, features_df):
        """创建交互特征"""
        
        # 情绪与价格的交互
        if all(m in features_df.columns for m in ['social_sentiment_composite', 'return_1d']):
            features_df['sentiment_price_interaction'] = (
                features_df['social_sentiment_composite'] * features_df['return_1d']
            )
        
        # 链上流与衍生品的交互
        if all(m in features_df.columns for m in ['net_exchange_flow', 'funding_rate']):
            features_df['flow_funding_interaction'] = (
                features_df['net_exchange_flow'] * features_df['funding_rate']
            )
        
        # 成交量与波动率的交互
        if all(m in features_df.columns for m in ['volume_ratio_20d', 'volatility_20d']):
            features_df['volume_vol_interaction'] = (
                features_df['volume_ratio_20d'] * features_df['volatility_20d']
            )
        
        return features_df
    
    def _create_time_features(self, features_df):
        """创建时间特征"""
        
        if not features_df.empty:
            # 周期性特征
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_month'] = features_df.index.day
            features_df['month'] = features_df.index.month
            features_df['week_of_year'] = features_df.index.isocalendar().week
            
            # 市场时间特征
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            features_df['is_month_end'] = (features_df.index.is_month_end).astype(int)
            
            # 季节性编码
            features_df['sin_day'] = np.sin(2 * np.pi * features_df.index.dayofyear / 365)
            features_df['cos_day'] = np.cos(2 * np.pi * features_df.index.dayofyear / 365)
        
        return features_df
    
    def _create_market_regime_features(self, features_df, aligned_df):
        """创建市场状态特征"""
        
        if 'close' in aligned_df.columns:
            # 趋势状态
            ma_20 = aligned_df['close'].rolling(20).mean()
            ma_50 = aligned_df['close'].rolling(50).mean()
            
            features_df['trend_strength'] = (ma_20 - ma_50) / ma_50
            features_df['is_uptrend'] = (ma_20 > ma_50).astype(int)
            features_df['is_downtrend'] = (ma_20 < ma_50).astype(int)
            
            # 波动率状态
            volatility = aligned_df['close'].pct_change().rolling(20).std()
            volatility_rank = volatility.rank(pct=True)
            
            features_df['volatility_regime'] = pd.cut(
                volatility_rank,
                bins=[0, 0.3, 0.7, 1],
                labels=['low', 'medium', 'high']
            ).astype(str)
            
            features_df['is_high_vol'] = (volatility_rank > 0.7).astype(int)
            features_df['is_low_vol'] = (volatility_rank < 0.3).astype(int)
            
            # 市场情绪状态
            if 'social_sentiment_composite' in features_df.columns:
                sentiment = features_df['social_sentiment_composite']
                sentiment_rank = sentiment.rank(pct=True)
                
                features_df['is_bullish_sentiment'] = (sentiment_rank > 0.7).astype(int)
                features_df['is_bearish_sentiment'] = (sentiment_rank < 0.3).astype(int)
        
        return features_df
    
    def create_labels(self, price_df: pd.DataFrame, horizon: int = 5, threshold: float = 0.03) -> pd.Series:
        """创建标签"""
        
        try:
            if len(price_df) < horizon + 1:
                return pd.Series([0] * len(price_df), index=price_df.index)
            
            # 计算未来收益
            future_returns = price_df['close'].pct_change(horizon).shift(-horizon)
            
            # 创建三分类标签
            labels = pd.Series(0, index=price_df.index)  # 0 = 横盘
            
            # 只对有效数据点赋值
            valid_mask = ~future_returns.isna()
            valid_returns = future_returns[valid_mask]
            
            # 上涨标签
            labels.loc[valid_mask & (valid_returns > threshold)] = 1
            
            # 下跌标签
            labels.loc[valid_mask & (valid_returns < -threshold)] = -1
            
            # 统计分布
            label_counts = labels.value_counts()
            logger.info(f"标签分布: {dict(label_counts)}")
            
            return labels
            
        except Exception as e:
            logger.error(f"创建标签失败: {e}")
            return pd.Series([0] * len(price_df), index=price_df.index)

# ==================== 改进的交易引擎（考虑手续费） ====================

class ImprovedTradingEngine:
    """改进的交易引擎 - 精确手续费计算"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trading_config = config.trading_config
        
        # 交易状态
        self.positions = {
            'DOGEUSDT': {
                'quantity': 0.0,
                'entry_price': 0.0,
                'entry_time': None,
                'unrealized_pnl': 0.0
            }
        }
        
        self.balance = self.trading_config['initial_balance']
        self.trade_history = deque(maxlen=1000)  # 使用deque限制大小
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # 风险控制
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.daily_trade_count = 0
        
        # 手续费统计
        self.total_commission = 0.0
        
        # 线程锁
        self.trade_lock = threading.RLock()
        
        logger.info("改进的交易引擎初始化完成")
    
    def calculate_commission(self, trade_value: float) -> float:
        """计算手续费（考虑分层和最低手续费）"""
        
        # 获取基础手续费率
        base_rate = self.trading_config['commission_rate']
        
        # 应用分层费率
        commission_rate = base_rate
        for tier in sorted(self.trading_config['commission_tiers'], key=lambda x: x['threshold']):
            if trade_value >= tier['threshold']:
                commission_rate = tier['rate']
        
        # 计算手续费
        commission = trade_value * commission_rate
        
        # 应用最低手续费
        min_commission = self.trading_config['min_commission']
        if commission < min_commission:
            commission = min_commission
        
        return commission
    
    def execute_trade(self, signal: Dict, current_price: float) -> Dict:
        """执行交易 - 精确计算手续费"""
        
        with self.trade_lock:
            trade_result = {
                'timestamp': datetime.now(),
                'symbol': 'DOGEUSDT',
                'action': signal.get('action', 'HOLD'),
                'signal_strength': signal.get('strength', 'NEUTRAL'),
                'confidence': signal.get('confidence', 0.5),
                'quantity': 0.0,
                'price': current_price,
                'value': 0.0,
                'commission': 0.0,
                'pnl': 0.0,
                'pnl_ratio': 0.0,
                'success': False,
                'reason': '',
                'position_after': self.positions['DOGEUSDT']['quantity'],
                'balance_after': self.balance
            }
            
            try:
                # 检查风险控制
                risk_check = self._check_risk_controls(signal, current_price)
                if not risk_check['allowed']:
                    trade_result['reason'] = risk_check['reason']
                    return trade_result
                
                action = signal['action']
                position = self.positions['DOGEUSDT']
                
                if action == 'BUY':
                    trade_result = self._execute_buy_with_commission(trade_result, signal, current_price)
                    
                elif action == 'SELL':
                    trade_result = self._execute_sell_with_commission(trade_result, signal, current_price)
                    
                else:  # HOLD
                    trade_result['reason'] = '持有信号，不交易'
                
                # 更新交易历史
                if trade_result['success']:
                    self.trade_history.append(trade_result.copy())
                    self._update_performance_metrics(trade_result)
                
                return trade_result
                
            except Exception as e:
                logger.error(f"交易执行失败: {e}")
                trade_result['reason'] = f'交易执行错误: {str(e)}'
                return trade_result
    
    def _check_risk_controls(self, signal: Dict, current_price: float) -> Dict:
        """风险控制检查"""
        
        # 检查最大仓位限制
        max_position_value = self.balance * self.trading_config['max_position']
        position_value = self.positions['DOGEUSDT']['quantity'] * current_price
        
        if signal['action'] == 'BUY':
            proposed_trade_value = signal.get('position_size', 0) * current_price
            # 预估手续费
            estimated_commission = self.calculate_commission(proposed_trade_value)
            
            if position_value + proposed_trade_value + estimated_commission > max_position_value:
                return {
                    'allowed': False,
                    'reason': f'超出最大仓位限制: {max_position_value:.2f} USDT'
                }
            
            # 检查资金是否足够（包含手续费）
            if proposed_trade_value + estimated_commission > self.balance:
                return {
                    'allowed': False,
                    'reason': f'资金不足，需要 {proposed_trade_value+estimated_commission:.2f}，可用 {self.balance:.2f}'
                }
        
        # 检查连续亏损
        max_consecutive_losses = 5
        if self.consecutive_losses >= max_consecutive_losses:
            return {
                'allowed': False,
                'reason': f'连续亏损{self.consecutive_losses}次，暂停交易'
            }
        
        # 检查日交易次数限制
        max_daily_trades = 20
        if self.daily_trade_count >= max_daily_trades:
            return {
                'allowed': False,
                'reason': f'达到日交易次数限制: {max_daily_trades}次'
            }
        
        # 检查最大回撤
        max_allowed_drawdown = 0.1  # 10%
        if self.max_drawdown >= max_allowed_drawdown:
            return {
                'allowed': False,
                'reason': f'达到最大回撤限制: {max_allowed_drawdown:.1%}'
            }
        
        return {'allowed': True, 'reason': ''}
    
    def _execute_buy_with_commission(self, trade_result: Dict, signal: Dict, current_price: float) -> Dict:
        """执行买入 - 精确计算手续费"""
        
        position = self.positions['DOGEUSDT']
        position_size = signal.get('position_size', 0)
        
        if position_size <= 0:
            trade_result['reason'] = '无效的仓位大小'
            return trade_result
        
        # 计算交易价值
        trade_value = position_size * current_price
        
        # 计算手续费
        commission = self.calculate_commission(trade_value)
        self.total_commission += commission
        
        # 检查资金是否足够（包含手续费）
        if trade_value + commission > self.balance:
            trade_result['reason'] = f'资金不足，需要 {trade_value+commission:.2f}，可用 {self.balance:.2f}'
            return trade_result
        
        # 执行买入
        old_quantity = position['quantity']
        old_value = old_quantity * position['entry_price'] if old_quantity > 0 else 0
        
        new_quantity = old_quantity + position_size
        new_value = old_value + trade_value
        
        # 计算平均入场价（包含手续费成本）
        if new_quantity > 0:
            # 入场价 = (总成本) / 数量
            # 总成本 = 交易价值 + 手续费
            new_entry_price = (new_value + commission) / new_quantity
        else:
            new_entry_price = 0
        
        # 更新仓位
        position['quantity'] = new_quantity
        position['entry_price'] = new_entry_price
        position['entry_time'] = datetime.now()
        
        # 更新资金
        self.balance -= (trade_value + commission)
        
        # 更新交易结果
        trade_result.update({
            'quantity': position_size,
            'value': trade_value,
            'commission': commission,
            'success': True,
            'reason': f'买入 {position_size:.0f} DOGE @ {current_price:.6f}，手续费: {commission:.4f} USDT',
            'position_after': new_quantity,
            'balance_after': self.balance,
            'entry_price': new_entry_price
        })
        
        logger.info(f"买入执行: {position_size:.0f} DOGE @ {current_price:.6f}，手续费: {commission:.4f}")
        
        return trade_result
    
    def _execute_sell_with_commission(self, trade_result: Dict, signal: Dict, current_price: float) -> Dict:
        """执行卖出 - 精确计算手续费"""
        
        position = self.positions['DOGEUSDT']
        position_size = signal.get('position_size', 0)
        
        if position_size <= 0:
            trade_result['reason'] = '无效的仓位大小'
            return trade_result
        
        if position_size > position['quantity']:
            position_size = position['quantity']  # 调整为最大可卖出数量
        
        # 计算交易价值
        trade_value = position_size * current_price
        
        # 计算手续费
        commission = self.calculate_commission(trade_value)
        self.total_commission += commission
        
        # 计算盈亏（基于包含手续费的入场价）
        entry_price = position['entry_price'] if position['quantity'] > 0 else current_price
        pnl = (current_price - entry_price) * position_size - commission
        pnl_ratio = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # 执行卖出
        position['quantity'] -= position_size
        
        # 如果全部卖出，重置入场价
        if position['quantity'] == 0:
            position['entry_price'] = 0
            position['entry_time'] = None
        
        # 更新资金
        self.balance += (trade_value - commission)
        
        # 更新交易结果
        trade_result.update({
            'quantity': position_size,
            'value': trade_value,
            'commission': commission,
            'pnl': pnl,
            'pnl_ratio': pnl_ratio,
            'success': True,
            'reason': f'卖出 {position_size:.0f} DOGE @ {current_price:.6f}, '
                     f'盈亏: {pnl:.2f} USDT ({pnl_ratio:.2%}), '
                     f'手续费: {commission:.4f} USDT',
            'position_after': position['quantity'],
            'balance_after': self.balance
        })
        
        logger.info(f"卖出执行: {position_size:.0f} DOGE @ {current_price:.6f}, "
                   f"盈亏: {pnl:.2f} USDT, 手续费: {commission:.4f}")
        
        return trade_result
    
    def _update_performance_metrics(self, trade_result: Dict):
        """更新性能指标"""
        
        # 更新日交易计数
        self.daily_trade_count += 1
        
        # 更新连续亏损计数
        if 'pnl' in trade_result:
            if trade_result['pnl'] < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # 更新总盈亏
            self.total_pnl += trade_result['pnl']
            self.daily_pnl += trade_result['pnl']
        
        # 计算当前回撤
        if self.total_pnl < 0:
            current_drawdown = abs(self.total_pnl) / self.trading_config['initial_balance']
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_position_summary(self) -> Dict:
        """获取仓位摘要"""
        
        with self.trade_lock:
            position = self.positions['DOGEUSDT']
            current_price = 0.08  # 模拟价格
            
            unrealized_pnl = 0.0
            if position['quantity'] > 0 and position['entry_price'] > 0:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            
            position_value = position['quantity'] * current_price
            
            return {
                'symbol': 'DOGEUSDT',
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'position_value': position_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_ratio': unrealized_pnl / (position['entry_price'] * position['quantity']) 
                                       if position['entry_price'] * position['quantity'] > 0 else 0,
                'balance': self.balance,
                'total_value': self.balance + position_value,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'total_commission': self.total_commission,
                'max_drawdown': self.max_drawdown,
                'consecutive_losses': self.consecutive_losses,
                'daily_trade_count': self.daily_trade_count
            }
    
    def reset_daily_metrics(self):
        """重置日度指标"""
        with self.trade_lock:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            logger.info("日度指标已重置")

# ==================== 稳定性改进的GUI ====================

class StableTradingGUI:
    """稳定性的交易系统图形界面"""
    
    def __init__(self, root, data_manager, model_manager, trading_engine):
        self.root = root
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.trading_engine = trading_engine
        self.simple_trader = None
        
        # 系统状态
        self.is_running = False
        self.is_training = False
        self.use_simple_trader = tk.BooleanVar(value=False)
        
        # 控制变量
        self.update_id = None
        self.trading_loop_id = None
        
        # 数据存储（使用deque限制大小）
        self.signals_history = deque(maxlen=500)
        self.trades_history = deque(maxlen=500)
        
        # 错误计数器
        self.error_count = 0
        self.max_errors = 10
        
        # 更新间隔
        self.update_interval = 5000  # 5秒
        self.trading_interval = 30000  # 30秒
        
        # 设置窗口
        self.root.title("DOGE多因子量化交易系统 - 稳定版")
        self.root.geometry("1400x900")
        
        # 设置样式
        self.setup_styles()
        
        # 创建界面
        self.create_widgets()
        
        logger.info("稳定性GUI初始化完成")
        
        # 启动数据更新（但不启动交易循环）
        self.schedule_update()
    
    def setup_styles(self):
        """设置界面样式"""
        self.colors = {
            'bg_dark': '#2c3e50',
            'bg_medium': '#34495e',
            'bg_light': '#ecf0f1',
            'text_light': '#ecf0f1',
            'text_dark': '#2c3e50',
            'primary': '#3498db',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#1abc9c'
        }
        
        # 字体
        self.font_title = tkFont.Font(family="Microsoft YaHei", size=16, weight="bold")
        self.font_subtitle = tkFont.Font(family="Microsoft YaHei", size=12, weight="bold")
        self.font_normal = tkFont.Font(family="Microsoft YaHei", size=10)
        self.font_small = tkFont.Font(family="Microsoft YaHei", size=9)
    
    def create_widgets(self):
        """创建界面组件"""
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部标题栏
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            title_frame,
            text="🐕 DOGE多因子量化交易系统 (稳定版)",
            font=self.font_title,
            foreground=self.colors['primary']
        ).pack(side=tk.LEFT)
        
        # 系统状态标签
        self.status_label = ttk.Label(
            title_frame,
            text="⚪ 系统就绪",
            font=self.font_normal,
            foreground=self.colors['success']
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # 创建两列布局
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 控制面板
        control_frame = ttk.LabelFrame(left_panel, text="系统控制", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # 启动按钮
        self.start_button = ttk.Button(
            button_frame,
            text="🚀 启动交易",
            command=self.start_trading_system,
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # 停止按钮
        self.stop_button = ttk.Button(
            button_frame,
            text="🛑 停止交易",
            command=self.stop_trading_system,
            width=15,
            state='disabled'
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 重启按钮
        self.restart_button = ttk.Button(
            button_frame,
            text="🔄 重启系统",
            command=self.restart_system,
            width=15
        )
        self.restart_button.pack(side=tk.LEFT, padx=5)

        # API配置
        api_frame = ttk.LabelFrame(control_frame, text="API配置", padding=8)
        api_frame.pack(fill=tk.X, pady=5)
        ttk.Label(api_frame, text="API Key").grid(row=0, column=0, sticky="w")
        ttk.Label(api_frame, text="API Secret").grid(row=1, column=0, sticky="w")
        ttk.Label(api_frame, text="代理").grid(row=2, column=0, sticky="w")
        self.api_key_entry = ttk.Entry(api_frame, width=28, show="*")
        self.api_secret_entry = ttk.Entry(api_frame, width=28, show="*")
        self.proxy_entry = ttk.Entry(api_frame, width=28)
        self.api_key_entry.grid(row=0, column=1, padx=4, pady=2)
        self.api_secret_entry.grid(row=1, column=1, padx=4, pady=2)
        self.proxy_entry.grid(row=2, column=1, padx=4, pady=2)
        self.test_api_button = ttk.Button(api_frame, text="测试连接", command=self.test_api_connection, width=12)
        self.test_api_button.grid(row=0, column=2, padx=4, pady=2, rowspan=2, sticky="ns")
        self.api_status_label = ttk.Label(api_frame, text="未测试", foreground=self.colors['warning'])
        self.api_status_label.grid(row=2, column=2, padx=4, pady=2)

        # 简化交易模式选择
        simple_frame = ttk.Frame(control_frame)
        simple_frame.pack(fill=tk.X, pady=4)
        simple_check = ttk.Checkbutton(simple_frame, text="启用简化自动交易 (MA+RSI)", variable=self.use_simple_trader)
        simple_check.pack(side=tk.LEFT)
        
        # 实时数据面板
        data_frame = ttk.LabelFrame(left_panel, text="实时数据", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建数据网格
        data_grid = ttk.Frame(data_frame)
        data_grid.pack(fill=tk.BOTH, expand=True)
        
        # 价格数据
        price_frame = ttk.LabelFrame(data_grid, text="价格数据", padding=10)
        price_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.price_labels = {}
        price_fields = [
            ("当前价格", "price", "${:.6f}"),
            ("24H变化", "change_24h", "{:.2%}"),
            ("成交量", "volume", "{:.0f}"),
            ("最高价", "high", "${:.6f}"),
            ("最低价", "low", "${:.6f}")
        ]
        
        for i, (label, key, fmt) in enumerate(price_fields):
            ttk.Label(price_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=2)
            self.price_labels[key] = ttk.Label(price_frame, text="--", font=self.font_normal)
            self.price_labels[key].grid(row=i, column=1, sticky="w", pady=2)
        
        # 市场情绪数据
        sentiment_frame = ttk.LabelFrame(data_grid, text="市场情绪", padding=10)
        sentiment_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.sentiment_labels = {}
        sentiment_fields = [
            ("Twitter情绪", "twitter", "{:.2f}"),
            ("Reddit情绪", "reddit", "{:.2f}"),
            ("新闻情绪", "news", "{:.2f}"),
            ("综合情绪", "composite", "{:.2f}")
        ]
        
        for i, (label, key, fmt) in enumerate(sentiment_fields):
            ttk.Label(sentiment_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=2)
            self.sentiment_labels[key] = ttk.Label(sentiment_frame, text="--", font=self.font_normal)
            self.sentiment_labels[key].grid(row=i, column=1, sticky="w", pady=2)
        
        # 仓位信息面板
        position_frame = ttk.LabelFrame(right_panel, text="仓位信息", padding=10)
        position_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 仓位数据网格
        position_grid = ttk.Frame(position_frame)
        position_grid.pack(fill=tk.X)
        
        self.position_labels = {}
        position_fields = [
            ("持仓数量", "quantity", "{:.0f} DOGE"),
            ("入场价格", "entry_price", "${:.6f}"),
            ("当前价格", "current_price", "${:.6f}"),
            ("浮动盈亏", "unrealized_pnl", "${:.2f} ({:.2%})"),
            ("账户余额", "balance", "${:.2f}"),
            ("总资产", "total_value", "${:.2f}"),
            ("总盈亏", "total_pnl", "${:.2f}"),
            ("总手续费", "total_commission", "${:.2f}")
        ]
        
        for i, (label, key, fmt) in enumerate(position_fields):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(position_grid, text=f"{label}:").grid(
                row=row, column=col, sticky="w", padx=5, pady=2
            )
            self.position_labels[key] = ttk.Label(
                position_grid, 
                text="--", 
                font=self.font_normal,
                foreground=self.colors['primary']
            )
            self.position_labels[key].grid(
                row=row, column=col+1, sticky="w", padx=5, pady=2
            )
        
        # 交易信号面板
        signal_frame = ttk.LabelFrame(right_panel, text="交易信号", padding=10)
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 信号显示区域
        self.signal_display = tk.Text(
            signal_frame,
            height=8,
            width=40,
            font=("Consolas", 10),
            bg='#2c3e50',
            fg='#ecf0f1',
            relief=tk.FLAT
        )
        self.signal_display.pack(fill=tk.BOTH, expand=True)
        self.signal_display.insert(tk.END, "等待交易信号...\n")
        self.signal_display.configure(state='disabled')
        
        # 系统日志面板
        log_frame = ttk.LabelFrame(right_panel, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 日志显示
        self.log_display = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Consolas", 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            relief=tk.FLAT
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)
        self.log_display.insert(tk.END, "系统日志\n")
        self.log_display.insert(tk.END, "="*40 + "\n")
        self.log_display.configure(state='disabled')
        
        # 底部状态栏
        self.status_bar = ttk.Label(
            self.root,
            text="就绪 | 最后更新: --",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_trading_system(self):
        """启动交易系统"""
        if not self.is_running:
            self.is_running = True
            self.error_count = 0
            if self.use_simple_trader.get():
                self.simple_trader = SimpleBinanceAutoTrader(
                    api_key=self.api_key_entry.get().strip(),
                    api_secret=self.api_secret_entry.get().strip(),
                    proxy=self.proxy_entry.get().strip(),
                    interval="5m",
                    lookback=120,
                    initial_balance=self.trading_engine.balance if hasattr(self.trading_engine, 'balance') else 1000.0,
                    live=True,  # 仅使用币安实盘数据
                    min_notional=5.0
                )
            
            self.status_label.configure(
                text="🟢 交易运行中",
                foreground=self.colors['success']
            )
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            
            # 启动数据流
            if self.data_manager:
                self.data_manager.start_data_streaming()
            
            # 启动交易循环
            self.schedule_trading_loop()
            
            self.log_message("交易系统已启动", "info")
    
    def stop_trading_system(self):
        """停止交易系统"""
        if self.is_running:
            self.is_running = False
            self.simple_trader = None
            
            # 停止定时器
            if self.update_id:
                self.root.after_cancel(self.update_id)
                self.update_id = None
            
            if self.trading_loop_id:
                self.root.after_cancel(self.trading_loop_id)
                self.trading_loop_id = None
            
            # 停止数据流
            if self.data_manager:
                self.data_manager.stop_data_streaming()
            
            self.status_label.configure(
                text="🔴 交易已停止",
                foreground=self.colors['danger']
            )
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            
            self.log_message("交易系统已停止", "info")
    
    def restart_system(self):
        """重启系统"""
        self.log_message("正在重启系统...", "warning")
        
        # 停止当前系统
        self.stop_trading_system()
        
        # 重置错误计数器
        self.error_count = 0
        
        # 重新启动
        self.root.after(1000, self.start_trading_system)
        
        self.log_message("系统重启完成", "success")
    
    def schedule_update(self):
        """计划GUI更新"""
        try:
            self.update_gui()
            # 使用after而不是递归，避免栈溢出
            self.update_id = self.root.after(self.update_interval, self.schedule_update)
        except Exception as e:
            self.log_message(f"GUI更新调度失败: {e}", "error")
            # 出错后重新调度
            self.update_id = self.root.after(5000, self.schedule_update)
    
    def schedule_trading_loop(self):
        """计划交易循环"""
        if self.is_running:
            try:
                if self.use_simple_trader.get() and self.simple_trader:
                    self.execute_simple_trading_cycle()
                else:
                    self.execute_trading_cycle()
                # 使用after而不是递归
                self.trading_loop_id = self.root.after(self.trading_interval, self.schedule_trading_loop)
            except Exception as e:
                self.log_message(f"交易循环调度失败: {e}", "error")
                self.error_count += 1
                
                if self.error_count >= self.max_errors:
                    self.log_message("错误过多，停止交易系统", "danger")
                    self.stop_trading_system()
                else:
                    # 出错后稍等再试
                    self.trading_loop_id = self.root.after(10000, self.schedule_trading_loop)
    
    def update_gui(self):
        """更新GUI显示"""
        try:
            # 更新时间戳
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.status_bar.configure(text=f"最后更新: {current_time}")
            
            # 更新价格数据
            if self.use_simple_trader.get() and self.simple_trader:
                latest_price = self.simple_trader.client.get_price(self.simple_trader.symbol) or 0.0
                self.price_labels['price'].configure(text=f"${latest_price:.6f}", foreground=self.colors['primary'])
                self.price_labels['high'].configure(text=f"${latest_price:.6f}")
                self.price_labels['low'].configure(text=f"${latest_price:.6f}")
                self.price_labels['change_24h'].configure(text="--", foreground=self.colors['warning'])
                self.price_labels['volume'].configure(text="--")
            elif self.data_manager:
                latest_data = self.data_manager.get_latest_data()
                
                if 'price' in latest_data and latest_data['price']:
                    price_data = latest_data['price']
                    
                    if 'close' in price_data:
                        current_price = price_data['close']
                        self.price_labels['price'].configure(
                            text=f"${current_price:.6f}",
                            foreground=self.colors['primary']
                        )
                    
                    # 模拟变化数据
                    change = np.random.uniform(-0.02, 0.02)
                    change_color = self.colors['success'] if change >= 0 else self.colors['danger']
                    self.price_labels['change_24h'].configure(
                        text=f"{change:+.2%}",
                        foreground=change_color
                    )
                    
                    # 其他价格数据
                    self.price_labels['volume'].configure(
                        text=f"{np.random.randint(1000000, 5000000):,}"
                    )
                    self.price_labels['high'].configure(
                        text=f"${current_price * 1.002:.6f}"
                    )
                    self.price_labels['low'].configure(
                        text=f"${current_price * 0.998:.6f}"
                    )
            
            # 使用简化交易时，不展示随机情绪
            if not (self.use_simple_trader.get() and self.simple_trader):
                twitter_sentiment = np.random.uniform(0.3, 0.8)
                reddit_sentiment = np.random.uniform(0.4, 0.9)
                news_sentiment = np.random.uniform(0.2, 0.7)
                composite_sentiment = (twitter_sentiment * 0.4 + 
                                     reddit_sentiment * 0.3 + 
                                     news_sentiment * 0.3)
                
                self.sentiment_labels['twitter'].configure(text=f"{twitter_sentiment:.2f}")
                self.sentiment_labels['reddit'].configure(text=f"{reddit_sentiment:.2f}")
                self.sentiment_labels['news'].configure(text=f"{news_sentiment:.2f}")
                self.sentiment_labels['composite'].configure(text=f"{composite_sentiment:.2f}")
            
            # 更新仓位信息
            if self.trading_engine:
                position_summary = self.trading_engine.get_position_summary()
                
                for key, label in self.position_labels.items():
                    if key in position_summary:
                        value = position_summary[key]
                        
                        # 格式化
                        if key in ['entry_price', 'current_price']:
                            fmt = "${:.6f}"
                        elif key in ['balance', 'total_value', 'total_pnl', 'unrealized_pnl', 'total_commission']:
                            fmt = "${:.2f}"
                        elif key == 'quantity':
                            fmt = "{:.0f} DOGE"
                        else:
                            fmt = "{}"
                        
                        # 处理特殊值
                        if key == 'unrealized_pnl' and isinstance(value, dict):
                            # 如果unrealized_pnl是字典，提取值
                            pnl_value = value.get('value', 0)
                            pnl_ratio = value.get('ratio', 0)
                            text = f"${pnl_value:.2f} ({pnl_ratio:.2%})"
                        elif key == 'unrealized_pnl' and isinstance(value, (int, float)):
                            # 如果是数字，需要计算比例
                            position_value = position_summary.get('position_value', 0)
                            entry_value = position_summary.get('entry_price', 0) * position_summary.get('quantity', 0)
                            ratio = value / entry_value if entry_value > 0 else 0
                            text = f"${value:.2f} ({ratio:.2%})"
                        else:
                            text = fmt.format(value)
                        
                        label.configure(text=text)
                        
                        # 设置颜色
                        if key in ['total_pnl', 'unrealized_pnl']:
                            if isinstance(value, (int, float)):
                                color = self.colors['success'] if value >= 0 else self.colors['danger']
                                label.configure(foreground=color)

            # 使用简化交易时同步仓位信息
            if self.use_simple_trader.get() and self.simple_trader:
                bal = self.simple_trader.balance
                qty = self.simple_trader.position_qty
                entry = self.simple_trader.entry_price
                last_price = self.simple_trader.client.get_price(self.simple_trader.symbol) or entry
                position_value = qty * last_price
                self.position_labels.get('balance', ttk.Label()).configure(text=f"${bal:.2f}")
                self.position_labels.get('quantity', ttk.Label()).configure(text=f"{qty:.0f} DOGE")
                self.position_labels.get('entry_price', ttk.Label()).configure(text=f"${entry:.6f}")
                self.position_labels.get('current_price', ttk.Label()).configure(text=f"${last_price:.6f}")
                self.position_labels.get('position_value', ttk.Label()).configure(text=f"${position_value:.2f}")
            
        except Exception as e:
            self.log_message(f"GUI更新失败: {e}", "error")
    
    def execute_trading_cycle(self):
        """执行交易周期"""
        try:
            # 生成交易信号
            signal = self.generate_trading_signal()
            
            # 显示信号
            self.display_trading_signal(signal)
            
            # 执行交易（模拟）
            if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                self.execute_trade(signal)
            
            self.log_message(f"交易周期完成: {signal['action']}，置信度: {signal['confidence']:.1%}", "info")
            
        except Exception as e:
            self.log_message(f"交易周期失败: {e}", "error")

    def execute_simple_trading_cycle(self):
        """执行简化交易周期 (使用Binance接口)"""
        try:
            result = self.simple_trader.run_cycle()
            signal = result.get('signal', {})
            trade = result.get('trade', {})
            account = result.get('account', {})
            self.log_message(f"[简化] 信号: {signal.get('action')} 置信度:{signal.get('confidence',0):.2f}", "info")
            if trade.get('success'):
                pnl = trade.get('pnl', 0)
                if pnl != 0:
                    self.log_message(f"[简化] 交易完成 PnL={pnl:.4f}", "success" if pnl >= 0 else "danger")
                else:
                    self.log_message(f"[简化] 交易完成", "success")
            else:
                err = trade.get('error', '交易未执行')
                self.log_message(f"[简化] 交易未执行: {err}", "warning")
            if account:
                self.position_labels.get('balance', ttk.Label()).configure(text=f"${account.get('balance',0):.2f}")
                self.position_labels.get('quantity', ttk.Label()).configure(text=f"{account.get('position_qty',0):.0f} DOGE")
                self.position_labels.get('entry_price', ttk.Label()).configure(text=f"${account.get('entry_price',0):.6f}")
                self.position_labels.get('current_price', ttk.Label()).configure(text=f"${account.get('last_price',0):.6f}")
        except Exception as e:
            self.log_message(f"简化交易周期失败: {e}", "error")
    
    def generate_trading_signal(self):
        """生成交易信号"""
        try:
            # 默认持有，交由简化交易器处理实盘信号
            return {
                'timestamp': datetime.now(),
                'action': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 0.5,
                'position_size': 0,
                'reasoning': ['简化实盘模式由Binance数据驱动']
            }
        except Exception as e:
            self.log_message(f"生成交易信号失败: {e}", "error")
            # 返回默认信号
            return {
                'timestamp': datetime.now(),
                'action': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 0.5,
                'position_size': 0,
                'reasoning': ['系统错误，默认持有']
            }

    def test_api_connection(self):
        """测试API连接并更新状态"""
        api_key = self.api_key_entry.get().strip()
        api_secret = self.api_secret_entry.get().strip()
        proxy = self.proxy_entry.get().strip()
        client = BinanceClient(api_key, api_secret, proxy)
        ok = client.test_connection()
        if ok:
            self.api_status_label.configure(text="连接正常", foreground=self.colors['success'])
            self.log_message("API连接成功", "success")
        else:
            self.api_status_label.configure(text="连接失败", foreground=self.colors['danger'])
            self.log_message("API连接失败，请检查配置", "danger")
    
    def display_trading_signal(self, signal: Dict):
        """显示交易信号"""
        try:
            self.signal_display.configure(state='normal')
            self.signal_display.delete(1.0, tk.END)
            
            # 格式化信号信息
            signal_text = f"时间: {signal['timestamp'].strftime('%H:%M:%S')}\n"
            signal_text += f"信号: {signal['action']} ({signal['strength']})\n"
            signal_text += f"置信度: {signal['confidence']:.1%}\n"
            
            if signal['position_size'] > 0:
                signal_text += f"建议仓位: {signal['position_size']:.0f} DOGE\n"
            
            signal_text += f"\n理由:\n"
            for reason in signal['reasoning']:
                signal_text += f"  • {reason}\n"
            
            signal_text += "\n" + "="*40 + "\n"
            
            # 设置颜色
            if signal['action'] == 'BUY':
                color = self.colors['success']
            elif signal['action'] == 'SELL':
                color = self.colors['danger']
            else:
                color = self.colors['warning']
            
            # 插入文本
            self.signal_display.insert(tk.END, signal_text)
            self.signal_display.tag_add("signal", "1.0", "end")
            self.signal_display.tag_config("signal", foreground=color)
            self.signal_display.configure(state='disabled')
            
            # 滚动到底部
            self.signal_display.see(tk.END)
            
        except Exception as e:
            self.log_message(f"显示交易信号失败: {e}", "error")
    
    def execute_trade(self, signal: Dict):
        """执行交易"""
        try:
            # 获取当前价格
            if self.data_manager:
                latest_data = self.data_manager.get_latest_data()
                current_price = latest_data['price'].get('close', 0.08) if 'price' in latest_data else 0.08
            else:
                current_price = 0.08
            
            # 执行交易
            if self.trading_engine:
                trade_result = self.trading_engine.execute_trade(signal, current_price)
                
                if trade_result['success']:
                    self.trades_history.append(trade_result)
                    self.log_message(
                        f"交易执行: {trade_result['reason']}",
                        "success" if 'pnl' not in trade_result or trade_result['pnl'] >= 0 else "danger"
                    )
                else:
                    self.log_message(f"交易失败: {trade_result['reason']}", "warning")
            
        except Exception as e:
            self.log_message(f"执行交易失败: {e}", "error")
    
    def log_message(self, message: str, level: str = "info"):
        """记录日志消息"""
        try:
            # 设置颜色
            colors = {
                'info': self.colors['primary'],
                'success': self.colors['success'],
                'warning': self.colors['warning'],
                'danger': self.colors['danger'],
                'error': self.colors['danger']
            }
            
            color = colors.get(level, self.colors['primary'])
            
            # 格式化消息
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}\n"
            
            # 更新日志显示
            self.log_display.configure(state='normal')
            self.log_display.insert(tk.END, formatted_message)
            
            # 应用颜色
            start_index = self.log_display.index("end-1c linestart")
            end_index = self.log_display.index("end-1c")
            self.log_display.tag_add(f"log_{level}", start_index, end_index)
            self.log_display.tag_config(f"log_{level}", foreground=color)
            
            # 限制日志行数
            lines = int(self.log_display.index('end-1c').split('.')[0])
            if lines > 200:
                self.log_display.delete(1.0, f"{lines-150}.0")
            
            self.log_display.see(tk.END)
            self.log_display.configure(state='disabled')
            
            # 同时输出到控制台
            print(f"[{level.upper()}] {message}")
            
        except Exception as e:
            print(f"日志记录失败: {e}")

# ==================== 简化的模型管理器 ====================

class SimplifiedModelManager:
    """简化的模型管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.models = {}
        self.model_performance = {}
        self.online_learning = config.model_config['online_learning']
        self.training_history = []
        
    def initialize_models(self, feature_dim: int):
        """初始化所有模型"""
        logger.info("初始化集成模型...")
        
        try:
            # XGBoost
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,  # 减少树的数量以提高速度
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            logger.info("XGBoost 模型初始化完成")
            
            # LightGBM
            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            logger.info("LightGBM 模型初始化完成")
            
            # 随机森林
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            logger.info("随机森林 模型初始化完成")
            
            # 梯度提升树
            self.models['gbt'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            logger.info("梯度提升树 模型初始化完成")
            
            logger.info(f"共初始化 {len(self.models)} 个模型")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")

# ==================== 主控制器 ====================

class StableMainController:
    """稳定的系统主控制器"""
    
    def __init__(self):
        # 初始化配置
        self.config = SystemConfig()
        
        # 初始化组件
        self.data_manager = StableDataManager(self.config)
        self.feature_engineer = ImprovedFeatureEngineer()
        self.model_manager = SimplifiedModelManager(self.config)
        self.trading_engine = ImprovedTradingEngine(self.config)
        
        # 系统状态
        self.is_initialized = False
        
        logger.info("稳定的主控制器初始化完成")
    
    def initialize_system(self, gui):
        """初始化系统"""
        logger.info("开始系统初始化...")
        
        try:
            # 1. 加载历史数据
            logger.info("加载历史数据...")
            self.data_manager.fetch_all_historical_data()
            
            # 2. 准备训练数据
            logger.info("准备训练数据...")
            
            # 获取对齐的数据
            price_data = self.data_manager.historical_data['price']['1d']
            social_data = self.data_manager.historical_data['social']
            onchain_data = self.data_manager.historical_data['onchain']
            derivatives_data = self.data_manager.historical_data['derivatives']
            
            # 创建特征
            features_df = self.feature_engineer.create_multi_factor_features(
                price_data, social_data, onchain_data, derivatives_data
            )
            
            # 创建标签
            labels = self.feature_engineer.create_labels(price_data)
            
            # 3. 初始化模型
            logger.info("初始化模型...")
            self.model_manager.initialize_models(features_df.shape[1])
            
            self.is_initialized = True
            
            # 更新GUI状态
            if gui:
                gui.log_message("系统初始化完成!", "success")
                gui.status_label.configure(
                    text="🟢 系统就绪",
                    foreground=gui.colors['success']
                )
            
            logger.info("系统初始化完成!")
            
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            if gui:
                gui.log_message(f"系统初始化失败: {e}", "danger")
            return False
    
    def stop_system(self):
        """停止系统"""
        # 停止数据流
        if self.data_manager:
            self.data_manager.stop_data_streaming()
        
        logger.info("系统已停止")

# ==================== 币安简化接口 ====================

class BinanceClient:
    """币安现货API简化封装"""
    
    def __init__(self, api_key="", api_secret="", proxy=""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.testnet_url = "https://testnet.binance.vision"
        self.proxies = {'https': proxy, 'http': proxy} if proxy else None
        self.timeout = 15
        self.recv_window = 5000
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'X-MBX-APIKEY': self.api_key
        })
    
    def _get_timestamp(self):
        return int(time.time() * 1000)
    
    def _sign(self, params):
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method, endpoint, params=None, signed=False, testnet=False):
        url = (self.testnet_url if testnet else self.base_url) + endpoint
        if signed:
            params = params or {}
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = self.recv_window
            params['signature'] = self._sign(params)
        try:
            if method == 'GET':
                resp = self.session.get(url, params=params, proxies=self.proxies, timeout=self.timeout)
            elif method == 'POST':
                resp = self.session.post(url, data=params, proxies=self.proxies, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"币安API请求失败: {e}")
            return None
    
    def get_price(self, symbol="DOGEUSDT"):
        try:
            resp = requests.get(f"{self.base_url}/api/v3/ticker/price", params={"symbol": symbol}, timeout=5, proxies=self.proxies)
            data = resp.json()
            if isinstance(data, dict) and 'price' in data:
                return float(data['price'])
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
        return None
    
    def get_klines(self, symbol="DOGEUSDT", interval="5m", limit=200):
        try:
            data = self._request('GET', '/api/v3/klines', {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            })
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"获取K线失败: {e}")
            return pd.DataFrame()
    
    def send_order(self, symbol, side, quantity, order_type="MARKET"):
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            result = self._request('POST', '/api/v3/order', params, signed=True)
            return {'success': bool(result and result.get('orderId')), 'result': result}
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {'success': False, 'error': str(e)}

    def test_connection(self):
        """测试API连通性"""
        try:
            result = self._request('GET', '/api/v3/ping')
            return result == {}
        except Exception as e:
            logger.error(f"测试API连接失败: {e}")
            return False

    def get_balance(self):
        """获取账户余额"""
        try:
            result = self._request('GET', '/api/v3/account', signed=True)
            balances = {}
            if result and 'balances' in result:
                for b in result['balances']:
                    asset = b['asset']
                    free = float(b['free'])
                    locked = float(b['locked'])
                    if free > 0 or locked > 0:
                        balances[asset] = {'free': free, 'locked': locked, 'total': free + locked}
            return balances
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return {}


# ==================== 简化量化交易器 ====================

class SimpleBinanceAutoTrader:
    """基于deepseek简化逻辑 + 币安接口的DOGE自动交易"""
    
    def __init__(
        self,
        api_key="",
        api_secret="",
        proxy="",
        symbol="DOGEUSDT",
        interval="5m",
        lookback=120,
        initial_balance=1000.0,
        live=False,
        ma_short=12,
        ma_long=36,
        rsi_period=14,
        position_scale=0.1,
        min_qty=1.0,
        zero_guard=1e-9,
        min_notional=5.0
    ):
        self.client = BinanceClient(api_key, api_secret, proxy)
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.balance = initial_balance
        self.live = live
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.position_scale = position_scale
        self.min_qty = min_qty
        self.zero_guard = zero_guard
        self.min_notional = min_notional
        self.position_qty = 0.0
        self.entry_price = 0.0
    
    def _fetch_candles(self):
        df = self.client.get_klines(self.symbol, self.interval, self.lookback)
        if df is None or df.empty:
            raise RuntimeError("无法从币安获取K线数据")
        df = df[['close_time', 'close', 'volume']].copy()
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df.set_index('close_time')
    
    def _indicators(self, df):
        f = df.copy()
        f['ma_s'] = f['close'].rolling(self.ma_short).mean()
        f['ma_l'] = f['close'].rolling(self.ma_long).mean()
        delta = f['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean().clip(lower=self.zero_guard)
        rs = avg_gain / avg_loss
        f['rsi'] = 100 - (100 / (1 + rs))
        return f.dropna()
    
    def _signal(self, frame):
        latest = frame.iloc[-1]
        price = float(latest['close'])
        safe_price = price if price > 0 else self.zero_guard
        trend_gap = latest['ma_s'] - latest['ma_l']
        rsi = latest['rsi']
        action = "HOLD"
        if trend_gap > 0 and rsi < 70:
            action = "BUY"
        elif trend_gap < 0 and rsi > 30 and self.position_qty > 0:
            action = "SELL"
        confidence = max(0.05, min(0.95,
            abs(trend_gap) / safe_price * 0.6 + abs(rsi - 50) / 50 * 0.4
        ))
        target_value = self.balance * self.position_scale
        max_affordable = self.balance / safe_price if safe_price > 0 else 0
        qty = max(target_value / safe_price, self.min_qty) if action == "BUY" else self.position_qty
        qty = min(qty, max_affordable)
        return {'action': action, 'price': price, 'qty': qty, 'confidence': confidence}
    
    def _execute(self, signal):
        if signal['action'] == "BUY" and self.position_qty == 0 and signal['qty'] > 0:
            if signal['qty'] * signal['price'] < self.min_notional:
                return {'success': False, 'error': '最小名义金额不足'}
            result = {'success': True}
            if self.live:
                result = self.client.send_order(self.symbol, "BUY", signal['qty'], "MARKET")
            if result.get('success'):
                self.position_qty = signal['qty']
                self.entry_price = signal['price']
                self.balance = max(self.balance - self.position_qty * signal['price'], 0)
            return result
        if signal['action'] == "SELL" and self.position_qty > 0:
            result = {'success': True}
            if self.live:
                result = self.client.send_order(self.symbol, "SELL", self.position_qty, "MARKET")
            if result.get('success'):
                pnl = (signal['price'] - self.entry_price) * self.position_qty
                self.balance += self.position_qty * signal['price']
                self.position_qty = 0.0
                self.entry_price = 0.0
                result['pnl'] = pnl
            return result
        return {'success': False, 'error': 'no action'}
    
    def run_cycle(self):
        candles = self._fetch_candles()
        frame = self._indicators(candles)
        if frame.empty:
            return {'error': 'insufficient data'}
        signal = self._signal(frame)
        trade = self._execute(signal)
        balances = {}
        if self.live:
            balances = self.client.get_balance()
            if 'USDT' in balances:
                self.balance = balances['USDT']['free']
            if 'DOGE' in balances:
                self.position_qty = balances['DOGE']['total']
        return {
            'signal': signal,
            'trade': trade,
            'account': {
                'balance': self.balance,
                'position_qty': self.position_qty,
                'entry_price': self.entry_price,
                'last_price': signal['price']
            }
        }

# ==================== 应用程序入口 ====================

def stable_main():
    """稳定的应用程序主函数"""
    
    print("=" * 60)
    print("DOGE多因子量化交易系统 - 稳定性优化版")
    print("=" * 60)
    print("改进特性:")
    print("  ✓ 保持所有原始功能和数据")
    print("  ✓ 精确手续费计算（分层费率 + 最低手续费）")
    print("  ✓ 线程安全和资源管理优化")
    print("  ✓ 防止GUI卡顿和运行暂停")
    print("=" * 60)
    
    try:
        # 创建主控制器
        controller = StableMainController()
        
        # 创建GUI
        root = tk.Tk()
        
        # 设置窗口
        root.title("DOGE多因子量化交易系统 - 稳定版")
        
        # 创建GUI实例
        app = StableTradingGUI(
            root,
            controller.data_manager,
            controller.model_manager,
            controller.trading_engine
        )
        
        # 在单独线程中初始化系统
        def initialize_system_async():
            success = controller.initialize_system(app)
            if not success:
                app.log_message("系统初始化失败!", "danger")
        
        # 延迟启动初始化
        root.after(1000, initialize_system_async)
        
        # 设置窗口关闭事件
        def on_closing():
            app.stop_trading_system()
            controller.stop_system()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 运行GUI主循环
        root.mainloop()
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        
        messagebox.showerror(
            "启动错误",
            f"应用程序启动失败:\n{str(e)}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true", help="运行无GUI的简化自动交易")
    parser.add_argument("--api-key", default="", help="币安API Key")
    parser.add_argument("--api-secret", default="", help="币安API Secret")
    parser.add_argument("--proxy", default="", help="HTTP/HTTPS 代理")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--live", action="store_true")
    args, _ = parser.parse_known_args()
    
    if args.simple:
        trader = SimpleBinanceAutoTrader(
            api_key=args.api_key,
            api_secret=args.api_secret,
            proxy=args.proxy,
            interval=args.interval,
            lookback=args.lookback,
            initial_balance=args.balance,
            live=args.live
        )
        result = trader.run_cycle()
        print(json.dumps(result, ensure_ascii=False, default=str))
    else:
        stable_main()
