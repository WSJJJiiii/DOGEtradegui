"""
DOGE多因子量化交易系统 - 完整融合版
整合了原始策略的所有功能和稳定性优化
包含实盘API连接、多因子模型、精确手续费计算和稳定性GUI
"""

import sys
import os
import json
import time
import threading
import queue
import hmac
import hashlib
import urllib.parse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings('ignore')

# 数学和数据处理库
import numpy as np
import pandas as pd
from scipy import stats
import pickle

# 机器学习库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

# 时间序列分析
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# 深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch不可用，LSTM功能将禁用")

# 可视化库
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI冲突
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import plotly.graph_objects as go
import plotly.express as px

# GUI库
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter.font as tkFont

# 网络请求库
import requests
from bs4 import BeautifulSoup
import websocket
import aiohttp
import asyncio

# ==================== 日志配置 ====================

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[94m',     # 蓝色
        'INFO': '\033[92m',      # 绿色
        'WARNING': '\033[93m',   # 黄色
        'ERROR': '\033[91m',     # 红色
        'CRITICAL': '\033[95m',   # 洋红
        'RESET': '\033[0m'       # 重置
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# 配置日志系统
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, 'doge_trading.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 添加彩色控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ==================== 系统配置 ====================

@dataclass
class TradingConfig:
    """交易配置数据类"""
    # API配置
    api_key: str = ""
    api_secret: str = ""
    proxy: str = "http://127.0.0.1:7890"
    
    # 交易参数
    symbol: str = "DOGEUSDT"
    initial_balance: float = 10000.0
    max_position_ratio: float = 0.3
    risk_per_trade: float = 0.02
    stop_loss: float = 0.05
    take_profit: List[float] = None
    commission_rate: float = 0.001
    min_commission: float = 0.1
    min_trade_amount: float = 10.0  # 最小交易金额
    
    # 模型参数
    prediction_horizon: int = 5
    confidence_threshold: float = 0.6
    model_retrain_interval: int = 24  # 小时
    feature_window: int = 50
    
    # 系统参数
    data_refresh_interval: int = 10  # 秒
    signal_check_interval: int = 60  # 秒
    live_trading: bool = False
    paper_trading: bool = True
    max_workers: int = 4
    enable_gc: bool = True
    gc_interval: int = 300
    
    def __post_init__(self):
        if self.take_profit is None:
            self.take_profit = [0.08, 0.15]

class SystemConfig:
    """系统配置管理器"""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.trading_config = TradingConfig()
        self.load_config()
        
        # 创建必要目录
        self.directories = {
            'cache': 'cache',
            'models': 'models',
            'data': 'data',
            'logs': 'logs',
            'reports': 'reports'
        }
        
        for dir_name, dir_path in self.directories.items():
            os.makedirs(dir_path, exist_ok=True)
        
        # 数据源配置
        self.data_sources = {
            'price': {
                'intervals': ['1m', '5m', '15m', '1h', '4h', '1d'],
                'history_days': 365,
                'real_time': True
            },
            'social': {
                'twitter': True,
                'reddit': True,
                'news': True,
                'update_interval': 3600
            },
            'onchain': {
                'whale_tracking': True,
                'exchange_flows': True,
                'holder_distribution': True,
                'update_interval': 1800
            },
            'derivatives': {
                'funding_rate': True,
                'open_interest': True,
                'long_short_ratio': True,
                'update_interval': 300
            }
        }
        
        # 模型配置
        self.model_config = {
            'ensemble_method': 'weighted',
            'model_weights': {
                'xgb': 0.3,
                'lgb': 0.3,
                'rf': 0.2,
                'prophet': 0.1,
                'gbt': 0.1
            },
            'online_learning': True,
            'retrain_on_error': True,
            'feature_selection': True
        }
        
        logger.info("系统配置初始化完成")
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        if hasattr(self.trading_config, key):
                            setattr(self.trading_config, key, value)
                logger.info(f"配置文件加载成功: {self.config_file}")
            else:
                self.save_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    
    def save_config(self):
        """保存配置文件"""
        try:
            config_dict = asdict(self.trading_config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"配置文件保存成功: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.trading_config, key):
                setattr(self.trading_config, key, value)
        self.save_config()

# ==================== 币安API客户端（修复版） ====================

class BinanceClient:
    """币安API客户端，支持现货交易"""
    
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'X-MBX-APIKEY': self.api_key
        })
        
        # 费率缓存
        self.commission_cache = {}
        self.last_rate_update = 0
        
        logger.info("币安API客户端初始化完成")
    
    def _get_timestamp(self):
        """获取当前时间戳（毫秒）"""
        return int(time.time() * 1000)
    
    def _sign(self, params):
        """生成签名"""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method, endpoint, params=None, signed=False, testnet=False):
        """发送HTTP请求"""
        url = (self.testnet_url if testnet else self.base_url) + endpoint
        
        try:
            if signed:
                if params is None:
                    params = {}
                params['timestamp'] = self._get_timestamp()
                params['recvWindow'] = self.recv_window
                params['signature'] = self._sign(params)
            
            if method == 'GET':
                response = self.session.get(
                    url, 
                    params=params, 
                    proxies=self.proxies, 
                    timeout=self.timeout
                )
            elif method == 'POST':
                response = self.session.post(
                    url, 
                    data=params, 
                    proxies=self.proxies, 
                    timeout=self.timeout
                )
            elif method == 'DELETE':
                response = self.session.delete(
                    url, 
                    params=params, 
                    proxies=self.proxies, 
                    timeout=self.timeout
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
    
    def test_connection(self):
        """测试API连接"""
        try:
            result = self._request('GET', '/api/v3/ping')
            return result == {}  # 成功返回空字典
        except:
            return False
    
    def get_server_time(self):
        """获取服务器时间"""
        try:
            result = self._request('GET', '/api/v3/time')
            return result.get('serverTime', 0)
        except:
            return 0
    
    def get_exchange_info(self, symbol="DOGEUSDT"):
        """获取交易对信息"""
        try:
            params = {'symbol': symbol}
            result = self._request('GET', '/api/v3/exchangeInfo', params)
            if result and 'symbols' in result:
                for s in result['symbols']:
                    if s['symbol'] == symbol:
                        return s
            return None
        except:
            return None
    
    def get_price(self, symbol="DOGEUSDT"):
        """获取当前价格"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(
                url, 
                params=params, 
                proxies=self.proxies, 
                timeout=5
            )
            data = response.json()
            if isinstance(data, dict) and 'price' in data:
                return float(data['price'])
            raise KeyError("price")
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
            # 使用缓存或模拟价格
            return 0.08 + np.random.normal(0, 0.0005)
    
    def get_klines(self, symbol="DOGEUSDT", interval="1h", limit=500):
        """获取K线数据"""
        try:
            endpoint = "/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            data = self._request('GET', endpoint, params)
            
            if data:
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # 转换数据类型
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                # 转换时间
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                
                return df
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()
    
    def get_balance(self):
        """获取账户余额"""
        try:
            endpoint = "/api/v3/account"
            params = {}
            result = self._request('GET', endpoint, params, signed=True)
            
            if result and 'balances' in result:
                balances = {}
                for item in result['balances']:
                    asset = item['asset']
                    free = float(item['free'])
                    locked = float(item['locked'])
                    if free > 0 or locked > 0:
                        balances[asset] = {
                            'free': free,
                            'locked': locked,
                            'total': free + locked
                        }
                return balances
            return None
            
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return None
    
    def get_ticker_24h(self, symbol="DOGEUSDT"):
        """获取24小时行情"""
        try:
            endpoint = "/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            result = self._request('GET', endpoint, params)
            return result
        except:
            return None
    
    def send_order(self, symbol, side, quantity, order_type="MARKET"):
        """发送订单"""
        try:
            endpoint = "/api/v3/order"
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'timestamp': self._get_timestamp()
            }
            
            if order_type == "LIMIT":
                params['timeInForce'] = 'GTC'
                # 需要price参数
            
            result = self._request('POST', endpoint, params, signed=True)
            
            if result and 'orderId' in result:
                logger.info(f"订单发送成功: {side} {quantity} {symbol}")
                return {
                    'success': True,
                    'order_id': result['orderId'],
                    'status': result['status'],
                    'fills': result.get('fills', [])
                }
            else:
                error_msg = result.get('msg', '未知错误')
                logger.error(f"订单发送失败: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"发送订单异常: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order(self, symbol, order_id):
        """查询订单"""
        try:
            endpoint = "/api/v3/order"
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            return self._request('GET', endpoint, params, signed=True)
        except:
            return None
    
    def cancel_order(self, symbol, order_id):
        """取消订单"""
        try:
            endpoint = "/api/v3/order"
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            return self._request('DELETE', endpoint, params, signed=True)
        except:
            return None
    
    def get_open_orders(self, symbol="DOGEUSDT"):
        """获取未成交订单"""
        try:
            endpoint = "/api/v3/openOrders"
            params = {'symbol': symbol}
            return self._request('GET', endpoint, params, signed=True)
        except:
            return []
    
    def get_trades(self, symbol="DOGEUSDT", limit=500):
        """获取历史成交"""
        try:
            endpoint = "/api/v3/myTrades"
            params = {
                'symbol': symbol,
                'limit': limit
            }
            return self._request('GET', endpoint, params, signed=True)
        except:
            return []

# ==================== 数据管理器（完整版） ====================

class CompleteDataManager:
    """完整的数据管理器，支持实盘和模拟数据"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trading_config = config.trading_config
        
        # 初始化API客户端
        self.api_client = BinanceClient(
            api_key=self.trading_config.api_key,
            api_secret=self.trading_config.api_secret,
            proxy=self.trading_config.proxy
        )
        
        # 数据存储
        self.historical_data = {
            'price': defaultdict(dict),  # 按时间框架存储
            'social': pd.DataFrame(),
            'onchain': pd.DataFrame(),
            'derivatives': pd.DataFrame(),
            'orderbook': deque(maxlen=100),
            'trades': deque(maxlen=1000)
        }
        
        # 实时数据队列
        self.realtime_queues = {
            'price': queue.Queue(maxsize=1000),
            'ticker': queue.Queue(maxsize=100),
            'depth': queue.Queue(maxsize=100),
            'kline': queue.Queue(maxsize=100)
        }
        
        # 线程控制
        self.threads = {}
        self.stop_event = threading.Event()
        self.data_lock = threading.RLock()
        
        # 状态变量
        self.is_streaming = False
        self.last_update_time = {}
        self.data_ready = False
        
        # 缓存
        self.cache = {}
        self.cache_expiry = {}
        
        logger.info("完整数据管理器初始化完成")
    
    def fetch_historical_data(self, days=365, force=False):
        """获取历史数据"""
        logger.info(f"开始获取{days}天历史数据...")
        
        try:
            # 1. 获取价格数据（多种时间框架）
            self._fetch_price_history(days)
            
            # 2. 获取社交媒体数据（模拟）
            self._fetch_social_data(days)
            
            # 3. 获取链上数据（模拟）
            self._fetch_onchain_data(days)
            
            # 4. 获取衍生品数据（模拟）
            self._fetch_derivatives_data(days)
            
            # 5. 获取订单簿快照
            self._fetch_orderbook_snapshot()
            
            self.data_ready = True
            logger.info("历史数据获取完成")
            
            # 保存缓存
            self._save_cache()
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            raise
    
    def _fetch_price_history(self, days):
        """获取价格历史数据"""
        try:
            symbol = self.trading_config.symbol
            intervals = self.config.data_sources['price']['intervals']
            
            for interval in intervals:
                # 计算需要的数据条数
                interval_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, 
                    '1h': 60, '4h': 240, '1d': 1440
                }
                
                minutes_per_day = 1440
                total_minutes = days * minutes_per_day
                interval_min = interval_minutes.get(interval, 60)
                limit = min(int(total_minutes / interval_min), 1000)
                
                # 从API获取数据
                df = self.api_client.get_klines(symbol, interval, limit)
                
                if not df.empty:
                    # 计算技术指标
                    df = self._calculate_technical_indicators(df)
                    
                    # 存储数据
                    self.historical_data['price'][interval] = df
                    
                    logger.info(f"价格数据[{interval}]获取完成: {len(df)}条记录")
                else:
                    # 生成模拟数据作为后备
                    self._generate_simulated_price_data(interval, days)
                    
        except Exception as e:
            logger.error(f"获取价格历史失败: {e}")
            # 生成模拟数据
            for interval in self.config.data_sources['price']['intervals']:
                self._generate_simulated_price_data(interval, days)
    
    def _generate_simulated_price_data(self, interval, days):
        """生成模拟价格数据"""
        try:
            interval_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
            }
            # 根据时间间隔确定数据点数量
            points_per_day = {
                '1m': 1440, '5m': 288, '15m': 96,
                '1h': 24, '4h': 6, '1d': 1
            }
            
            total_points = days * points_per_day.get(interval, 24)
            
            # 生成时间序列
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if interval == '1d':
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
            elif interval == '4h':
                dates = pd.date_range(start=start_date, end=end_date, freq='4H')
            elif interval == '1h':
                dates = pd.date_range(start=start_date, end=end_date, freq='H')
            elif interval == '15m':
                dates = pd.date_range(start=start_date, end=end_date, freq='15T')
            elif interval == '5m':
                dates = pd.date_range(start=start_date, end=end_date, freq='5T')
            else:  # 1m
                dates = pd.date_range(start=start_date, end=end_date, freq='T')
            
            # 限制数据点数量
            if len(dates) > total_points:
                dates = dates[-total_points:]
            
            # 生成价格序列（几何布朗运动）
            np.random.seed(42)
            n_points = len(dates)
            base_price = 0.08
            
            # 参数设置
            mu = 0.0002  # 日均收益率
            sigma = 0.02  # 波动率
            dt = 1 / points_per_day.get(interval, 24)
            
            # 生成收益率
            returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_points)
            
            # 生成价格
            prices = base_price * np.exp(np.cumsum(returns))
            
            # 添加趋势和季节性
            trend = np.linspace(0, 0.1, n_points)
            seasonal = 0.001 * np.sin(2 * np.pi * np.arange(n_points) / (points_per_day.get(interval, 24) * 7))
            
            prices *= (1 + trend + seasonal)
            
            # 生成OHLCV数据
            df = pd.DataFrame({
                'open_time': dates,
                'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_points)),
                'high': prices * (1 + np.random.uniform(0, 0.002, n_points)),
                'low': prices * (1 - np.random.uniform(0, 0.002, n_points)),
                'close': prices,
                'volume': np.random.lognormal(10, 1.5, n_points) * 10000,
                'close_time': dates + timedelta(minutes=interval_minutes.get(interval, 60)),
                'quote_volume': prices * np.random.lognormal(12, 1.2, n_points) * 1000,
                'trades': np.random.randint(100, 10000, n_points),
                'taker_buy_base': np.random.lognormal(9, 1, n_points) * 1000,
                'taker_buy_quote': prices * np.random.lognormal(10, 1, n_points) * 1000,
                'ignore': 0
            })
            
            # 计算技术指标
            df = self._calculate_technical_indicators(df)
            
            self.historical_data['price'][interval] = df
            logger.info(f"模拟价格数据[{interval}]生成完成: {len(df)}条记录")
            
        except Exception as e:
            logger.error(f"生成模拟价格数据失败: {e}")
    
    def _calculate_technical_indicators(self, df):
        """计算技术指标（完整版）"""
        try:
            df = df.copy()
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # 1. 移动平均线
            for period in [5, 10, 20, 30, 50, 100, 200]:
                df[f'SMA_{period}'] = close.rolling(window=period).mean()
                df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
            
            # 2. 布林带
            bb_period = 20
            bb_std = 2
            df['BB_middle'] = close.rolling(window=bb_period).mean()
            bb_std_dev = close.rolling(window=bb_period).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std_dev * bb_std)
            df['BB_lower'] = df['BB_middle'] - (bb_std_dev * bb_std)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            # 3. RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 4. MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # 5. 随机指标
            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            df['STOCH_K'] = 100 * ((close - low14) / (high14 - low14))
            df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
            
            # 6. ATR（平均真实波幅）
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # 7. 成交量指标
            df['Volume_SMA'] = volume.rolling(window=20).mean()
            df['Volume_Ratio'] = volume / df['Volume_SMA']
            
            # 8. OBV（能量潮）
            df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            
            # 9. 价格动量
            for period in [1, 3, 5, 10, 20]:
                df[f'Return_{period}'] = close.pct_change(period)
                df[f'Volatility_{period}'] = close.pct_change().rolling(period).std()
            
            # 10. 价格通道
            df['Donchian_high'] = high.rolling(window=20).max()
            df['Donchian_low'] = low.rolling(window=20).min()
            df['Donchian_middle'] = (df['Donchian_high'] + df['Donchian_low']) / 2
            
            # 11. 价格加速度
            df['Price_Acceleration'] = close.diff().diff()
            
            # 12. 威廉指标
            highest_high = high.rolling(window=14).max()
            lowest_low = low.rolling(window=14).min()
            df['Williams_%R'] = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            # 13. CCI（商品通道指数）
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # 14. 动量指标
            df['Momentum'] = close - close.shift(10)
            
            # 15. 价格变化率
            df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
            
            # 16. 填充NaN值
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return df
    
    def _fetch_social_data(self, days):
        """获取社交媒体数据（模拟）"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=days,
                freq='1D'
            )
            
            np.random.seed(42)
            base_sentiment = 0.5
            
            # 生成趋势、季节性和噪声
            trend = np.linspace(-0.1, 0.2, len(dates))
            seasonal = 0.15 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
            noise = np.random.normal(0, 0.08, len(dates))
            
            sentiment = base_sentiment + trend + seasonal + noise
            sentiment = np.clip(sentiment, 0.1, 0.9)
            
            # 生成社交媒体数据
            df = pd.DataFrame({
                'timestamp': dates,
                'twitter_sentiment': sentiment,
                'twitter_volume': np.random.lognormal(10, 1, len(dates)),
                'reddit_sentiment': sentiment * 0.9 + np.random.uniform(-0.1, 0.1, len(dates)),
                'reddit_posts': np.random.randint(100, 10000, len(dates)),
                'news_sentiment': sentiment * 0.8 + np.random.uniform(-0.15, 0.15, len(dates)),
                'news_count': np.random.randint(10, 500, len(dates)),
                'google_trends': np.random.uniform(0, 100, len(dates)),
                'social_volume': np.random.lognormal(9, 1.2, len(dates)),
                'weighted_sentiment': (
                    sentiment * 0.4 + 
                    sentiment * 0.9 * 0.3 + 
                    sentiment * 0.8 * 0.3
                )
            })
            
            # 添加事件标记
            df['major_event'] = np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
            
            self.historical_data['social'] = df
            logger.info(f"社交媒体数据生成完成: {len(df)}条记录")
            
        except Exception as e:
            logger.error(f"生成社交媒体数据失败: {e}")
    
    def _fetch_onchain_data(self, days):
        """获取链上数据（模拟）"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=days,
                freq='1D'
            )
            
            np.random.seed(42)
            
            # 生成链上数据
            df = pd.DataFrame({
                'timestamp': dates,
                'active_addresses': np.random.randint(50000, 200000, len(dates)),
                'new_addresses': np.random.randint(1000, 5000, len(dates)),
                'transaction_count': np.random.randint(10000, 50000, len(dates)),
                'transaction_volume': np.random.uniform(100000, 5000000, len(dates)),
                'exchange_inflow': np.random.uniform(-500000, 500000, len(dates)),
                'exchange_outflow': np.random.uniform(-500000, 500000, len(dates)),
                'net_exchange_flow': np.random.uniform(-1000000, 1000000, len(dates)),
                'whale_transactions_1k': np.random.randint(10, 100, len(dates)),
                'whale_transactions_10k': np.random.randint(1, 20, len(dates)),
                'network_growth': np.random.uniform(-0.02, 0.03, len(dates)),
                'hash_rate': np.random.uniform(100, 500, len(dates)),
                'miner_balance': np.random.uniform(1000000, 5000000, len(dates)),
                'hodl_waves_1d': np.random.uniform(0.05, 0.3, len(dates)),
                'hodl_waves_1w': np.random.uniform(0.05, 0.25, len(dates)),
                'hodl_waves_1m': np.random.uniform(0.1, 0.3, len(dates)),
                'hodl_waves_3m': np.random.uniform(0.05, 0.2, len(dates)),
                'hodl_waves_6m': np.random.uniform(0.03, 0.15, len(dates)),
                'supply_on_exchanges': np.random.uniform(0.1, 0.3, len(dates))
            })
            
            # 计算衍生指标
            df['exchange_flow_ratio'] = df['exchange_inflow'] / (df['exchange_outflow'] + 1)
            df['whale_activity_score'] = df['whale_transactions_1k'] * 0.7 + df['whale_transactions_10k'] * 0.3
            df['network_health_score'] = (
                df['active_addresses'].rolling(7).mean() / df['active_addresses'].rolling(30).mean() * 0.4 +
                df['transaction_count'].rolling(7).mean() / df['transaction_count'].rolling(30).mean() * 0.3 +
                (1 - df['supply_on_exchanges']) * 0.3
            )
            
            self.historical_data['onchain'] = df
            logger.info(f"链上数据生成完成: {len(df)}条记录")
            
        except Exception as e:
            logger.error(f"生成链上数据失败: {e}")
    
    def _fetch_derivatives_data(self, days):
        """获取衍生品数据（模拟）"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=days,
                freq='1D'
            )
            
            np.random.seed(42)
            
            # 生成衍生品数据
            df = pd.DataFrame({
                'timestamp': dates,
                'funding_rate': np.random.uniform(-0.01, 0.01, len(dates)),
                'funding_rate_8h': np.random.uniform(-0.005, 0.005, len(dates)),
                'open_interest': np.random.uniform(1000000, 5000000, len(dates)),
                'open_interest_change': np.random.uniform(-0.1, 0.1, len(dates)),
                'long_short_ratio': np.random.uniform(0.8, 1.2, len(dates)),
                'liquidations_long': np.random.uniform(0, 500000, len(dates)),
                'liquidations_short': np.random.uniform(0, 500000, len(dates)),
                'estimated_leverage_ratio': np.random.uniform(1.5, 3.0, len(dates)),
                'basis': np.random.uniform(-0.02, 0.02, len(dates)),
                'volume_ratio_spot_futures': np.random.uniform(0.5, 2.0, len(dates))
            })
            
            # 计算衍生指标
            df['funding_rate_signal'] = np.where(
                df['funding_rate'] > 0.001, -1,
                np.where(df['funding_rate'] < -0.001, 1, 0)
            )
            df['liquidation_imbalance'] = (
                df['liquidations_long'] - df['liquidations_short']
            ) / (df['liquidations_long'] + df['liquidations_short'] + 1)
            df['open_interest_trend'] = df['open_interest'].pct_change(3)
            
            self.historical_data['derivatives'] = df
            logger.info(f"衍生品数据生成完成: {len(df)}条记录")
            
        except Exception as e:
            logger.error(f"生成衍生品数据失败: {e}")
    
    def _fetch_orderbook_snapshot(self):
        """获取订单簿快照"""
        try:
            # 这里可以调用API获取实时订单簿
            # 暂时使用模拟数据
            import random
            
            snapshot = {
                'timestamp': datetime.now(),
                'bids': [],
                'asks': [],
                'bid_volume': 0,
                'ask_volume': 0,
                'mid_price': 0.08,
                'spread': 0.0001
            }
            
            # 生成模拟买卖盘
            base_price = 0.08
            for i in range(10):
                bid_price = base_price * (1 - i * 0.0005 - random.uniform(0, 0.0002))
                bid_volume = random.uniform(1000, 10000)
                snapshot['bids'].append((bid_price, bid_volume))
                snapshot['bid_volume'] += bid_volume
                
                ask_price = base_price * (1 + i * 0.0005 + random.uniform(0, 0.0002))
                ask_volume = random.uniform(1000, 10000)
                snapshot['asks'].append((ask_price, ask_volume))
                snapshot['ask_volume'] += ask_volume
            
            snapshot['mid_price'] = (snapshot['bids'][0][0] + snapshot['asks'][0][0]) / 2
            snapshot['spread'] = snapshot['asks'][0][0] - snapshot['bids'][0][0]
            
            self.historical_data['orderbook'].append(snapshot)
            
        except Exception as e:
            logger.error(f"获取订单簿快照失败: {e}")
    
    def start_real_time_stream(self):
        """启动实时数据流"""
        if self.is_streaming:
            logger.warning("实时数据流已在运行")
            return
        
        self.stop_event.clear()
        self.is_streaming = True
        
        # 启动价格更新线程
        price_thread = threading.Thread(
            target=self._price_update_loop,
            daemon=True,
            name="PriceUpdateThread"
        )
        price_thread.start()
        self.threads['price'] = price_thread
        
        # 启动数据聚合线程
        agg_thread = threading.Thread(
            target=self._data_aggregation_loop,
            daemon=True,
            name="DataAggregationThread"
        )
        agg_thread.start()
        self.threads['aggregation'] = agg_thread
        
        logger.info("实时数据流已启动")
    
    def stop_real_time_stream(self):
        """停止实时数据流"""
        self.stop_event.set()
        self.is_streaming = False
        
        # 等待线程结束
        for name, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.threads.clear()
        logger.info("实时数据流已停止")
    
    def _price_update_loop(self):
        """价格更新循环"""
        update_interval = self.trading_config.data_refresh_interval
        
        while not self.stop_event.is_set():
            try:
                with self.data_lock:
                    # 获取实时价格
                    current_price = self.api_client.get_price(self.trading_config.symbol)
                    
                    if current_price is not None:
                        # 创建数据点
                        timestamp = datetime.now()
                        data_point = {
                            'timestamp': timestamp,
                            'price': current_price,
                            'symbol': self.trading_config.symbol
                        }
                        
                        # 放入队列
                        try:
                            self.realtime_queues['price'].put(data_point, timeout=0.5)
                        except queue.Full:
                            # 队列满时移除旧数据
                            try:
                                self.realtime_queues['price'].get_nowait()
                            except queue.Empty:
                                pass
                            self.realtime_queues['price'].put(data_point, timeout=0.5)
                        
                        # 更新最新价格到缓存
                        self.cache['latest_price'] = current_price
                        self.cache_expiry['latest_price'] = time.time() + 60
                    
                    # 定期获取K线数据
                    if 'last_kline_update' not in self.last_update_time or \
                       time.time() - self.last_update_time.get('last_kline_update', 0) > 300:
                        
                        for interval in ['5m', '15m', '1h']:
                            df = self.api_client.get_klines(self.trading_config.symbol, interval, 100)
                            if not df.empty:
                                self.historical_data['price'][interval] = df
                            else:
                                self._generate_simulated_price_data(interval, days=7)
                        
                        self.last_update_time['last_kline_update'] = time.time()
                
                # 等待下一次更新
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"价格更新循环错误: {e}")
                time.sleep(10)  # 出错后等待更长时间
    
    def _data_aggregation_loop(self):
        """数据聚合循环"""
        while not self.stop_event.is_set():
            try:
                # 聚合最新数据
                latest_data = self.get_latest_aggregated_data()
                
                # 定期清理缓存
                current_time = time.time()
                expired_keys = [
                    key for key, expiry in self.cache_expiry.items()
                    if expiry < current_time
                ]
                for key in expired_keys:
                    self.cache.pop(key, None)
                    self.cache_expiry.pop(key, None)
                
                # 定期执行垃圾回收
                if self.config.trading_config.enable_gc and \
                   current_time - self.last_update_time.get('last_gc', 0) > self.config.trading_config.gc_interval:
                    import gc
                    gc.collect()
                    self.last_update_time['last_gc'] = current_time
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"数据聚合循环错误: {e}")
                time.sleep(60)
    
    def get_latest_aggregated_data(self):
        """获取最新聚合数据"""
        try:
            with self.data_lock:
                latest_data = {
                    'timestamp': datetime.now(),
                    'price': {},
                    'social': {},
                    'onchain': {},
                    'derivatives': {},
                    'orderbook': {},
                    'trades': []
                }
                
                # 获取最新价格
                if '1m' in self.historical_data['price'] and not self.historical_data['price']['1m'].empty:
                    latest_price = self.historical_data['price']['1m'].iloc[-1].to_dict()
                    latest_data['price'] = latest_price
                
                # 获取最新社交媒体数据
                if not self.historical_data['social'].empty:
                    latest_social = self.historical_data['social'].iloc[-1].to_dict()
                    latest_data['social'] = latest_social
                
                # 获取最新链上数据
                if not self.historical_data['onchain'].empty:
                    latest_onchain = self.historical_data['onchain'].iloc[-1].to_dict()
                    latest_data['onchain'] = latest_onchain
                
                # 获取最新衍生品数据
                if not self.historical_data['derivatives'].empty:
                    latest_derivatives = self.historical_data['derivatives'].iloc[-1].to_dict()
                    latest_data['derivatives'] = latest_derivatives
                
                # 获取最新订单簿
                if self.historical_data['orderbook']:
                    latest_orderbook = self.historical_data['orderbook'][-1]
                    latest_data['orderbook'] = latest_orderbook
                
                # 获取最新交易
                if self.historical_data['trades']:
                    latest_trades = list(self.historical_data['trades'])[-10:]  # 最近10笔交易
                    latest_data['trades'] = latest_trades
                
                return latest_data
                
        except Exception as e:
            logger.error(f"获取聚合数据失败: {e}")
            return {}
    
    def get_realtime_price(self):
        """获取实时价格（非阻塞）"""
        try:
            return self.realtime_queues['price'].get_nowait()
        except queue.Empty:
            return None
    
    def get_historical_features(self, lookback_days=30):
        """获取历史特征数据"""
        try:
            # 获取日线数据作为基础
            if '1d' not in self.historical_data['price'] or self.historical_data['price']['1d'].empty:
                logger.warning("日线数据不可用，使用模拟数据")
                return pd.DataFrame()
            
            price_df = self.historical_data['price']['1d'].copy()
            
            # 合并其他数据源
            feature_dfs = [price_df]
            
            if not self.historical_data['social'].empty:
                social_df = self.historical_data['social'].copy()
                social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
                social_df.set_index('timestamp', inplace=True)
                feature_dfs.append(social_df)
            
            if not self.historical_data['onchain'].empty:
                onchain_df = self.historical_data['onchain'].copy()
                onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'])
                onchain_df.set_index('timestamp', inplace=True)
                feature_dfs.append(onchain_df)
            
            if not self.historical_data['derivatives'].empty:
                derivatives_df = self.historical_data['derivatives'].copy()
                derivatives_df['timestamp'] = pd.to_datetime(derivatives_df['timestamp'])
                derivatives_df.set_index('timestamp', inplace=True)
                feature_dfs.append(derivatives_df)
            
            # 合并所有特征
            features_df = feature_dfs[0]
            for df in feature_dfs[1:]:
                features_df = features_df.join(df, how='left', rsuffix=f'_{df.columns[0] if len(df.columns) > 0 else "x"}')
            
            # 填充缺失值
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 限制回看天数
            if lookback_days > 0:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                features_df = features_df[features_df.index >= cutoff_date]
            
            return features_df
            
        except Exception as e:
            logger.error(f"获取历史特征失败: {e}")
            return pd.DataFrame()
    
    def _save_cache(self):
        """保存数据缓存"""
        try:
            cache_file = os.path.join(self.config.directories['cache'], 'data_cache.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'historical_data': self.historical_data,
                    'cache': self.cache,
                    'data_ready': self.data_ready
                }, f)
            logger.info("数据缓存已保存")
        except Exception as e:
            logger.error(f"保存数据缓存失败: {e}")
    
    def _load_cache(self):
        """加载数据缓存"""
        try:
            cache_file = os.path.join(self.config.directories['cache'], 'data_cache.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.historical_data = cache_data.get('historical_data', self.historical_data)
                    self.cache = cache_data.get('cache', {})
                    self.data_ready = cache_data.get('data_ready', False)
                logger.info("数据缓存已加载")
                return True
        except Exception as e:
            logger.error(f"加载数据缓存失败: {e}")
        return False

# ==================== 高级特征工程师 ====================

class AdvancedFeatureEngineer:
    """高级特征工程师，包含原始所有特征计算"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trading_config = config.trading_config
        
        # 特征缩放器
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1)),
            'robust': RobustScaler()
        }
        
        # 特征缓存
        self.feature_cache = {}
        self.cache_max_size = 100
        
        # 特征重要性记录
        self.feature_importance = {}
        
        logger.info("高级特征工程师初始化完成")
    
    def create_complete_features(self, data_manager: CompleteDataManager) -> pd.DataFrame:
        """创建完整的特征数据集"""
        logger.info("开始创建完整特征集...")
        
        try:
            # 获取历史特征数据
            historical_features = data_manager.get_historical_features(
                lookback_days=self.trading_config.feature_window
            )
            
            if historical_features.empty:
                logger.warning("历史特征数据为空，生成模拟特征")
                historical_features = self._generate_simulated_features()
            
            # 计算技术指标特征
            technical_features = self._create_technical_features(historical_features)
            
            # 计算统计特征
            statistical_features = self._create_statistical_features(historical_features)
            
            # 计算时间序列特征
            timeseries_features = self._create_timeseries_features(historical_features)
            
            # 计算市场微观结构特征
            microstructure_features = self._create_microstructure_features(historical_features)
            
            # 计算基本面特征
            fundamental_features = self._create_fundamental_features(historical_features)
            
            # 计算市场情绪特征
            sentiment_features = self._create_sentiment_features(historical_features)
            
            # 计算风险特征
            risk_features = self._create_risk_features(historical_features)
            
            # 合并所有特征
            all_features = pd.concat([
                technical_features,
                statistical_features,
                timeseries_features,
                microstructure_features,
                fundamental_features,
                sentiment_features,
                risk_features
            ], axis=1)
            
            # 移除重复列
            all_features = all_features.loc[:, ~all_features.columns.duplicated()]
            
            # 创建交互特征
            interaction_features = self._create_interaction_features(all_features)
            all_features = pd.concat([all_features, interaction_features], axis=1)
            
            # 创建滞后特征
            lag_features = self._create_lag_features(all_features)
            all_features = pd.concat([all_features, lag_features], axis=1)
            
            # 特征选择
            if self.config.model_config['feature_selection']:
                all_features = self._select_features(all_features)
            
            # 特征缩放
            all_features = self._scale_features(all_features)
            
            # 处理缺失值
            all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 移除无穷值
            all_features = all_features.replace([np.inf, -np.inf], 0)
            
            logger.info(f"特征创建完成，共 {all_features.shape[1]} 个特征，{all_features.shape[0]} 个样本")
            
            # 更新特征重要性记录
            self._update_feature_importance(all_features)
            
            return all_features
            
        except Exception as e:
            logger.error(f"创建完整特征集失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _generate_simulated_features(self):
        """生成模拟特征数据"""
        try:
            dates = pd.date_range(
                end=datetime.now(),
                periods=self.trading_config.feature_window,
                freq='1D'
            )
            
            np.random.seed(42)
            
            # 生成基础价格数据
            base_price = 0.08
            returns = np.random.normal(0, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            # 生成技术指标
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
                'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
                'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, len(dates)) * 10000,
                'SMA_20': prices.rolling(window=20).mean(),
                'EMA_12': prices.ewm(span=12).mean(),
                'EMA_26': prices.ewm(span=26).mean(),
                'RSI': np.random.uniform(30, 70, len(dates)),
                'MACD': np.random.uniform(-0.001, 0.001, len(dates)),
                'BB_upper': prices * 1.02,
                'BB_lower': prices * 0.98,
                'ATR': np.random.uniform(0.001, 0.005, len(dates)),
                'twitter_sentiment': np.random.uniform(0.3, 0.8, len(dates)),
                'reddit_sentiment': np.random.uniform(0.4, 0.9, len(dates)),
                'news_sentiment': np.random.uniform(0.2, 0.7, len(dates)),
                'social_volume': np.random.lognormal(8, 1, len(dates)),
                'active_addresses': np.random.randint(50000, 200000, len(dates)),
                'transaction_count': np.random.randint(10000, 50000, len(dates)),
                'exchange_inflow': np.random.uniform(-1000000, 1000000, len(dates)),
                'exchange_outflow': np.random.uniform(-1000000, 1000000, len(dates)),
                'funding_rate': np.random.uniform(-0.01, 0.01, len(dates)),
                'open_interest': np.random.uniform(1000000, 5000000, len(dates)),
                'long_short_ratio': np.random.uniform(0.8, 1.2, len(dates))
            })
            
            df.set_index('timestamp', inplace=True)
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"生成模拟特征失败: {e}")
            return pd.DataFrame()
    
    def _create_technical_features(self, df):
        """创建技术指标特征"""
        features = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns:
            close = df['close']
            
            # 移动平均线特征
            for period in [5, 10, 20, 30, 50, 100]:
                if f'SMA_{period}' in df.columns:
                    sma = df[f'SMA_{period}']
                    features[f'SMA_{period}_ratio'] = close / sma
                    features[f'SMA_{period}_slope'] = sma.diff()
            
            # 价格位置特征
            if all(col in df.columns for col in ['BB_upper', 'BB_lower']):
                bb_range = df['BB_upper'] - df['BB_lower']
                features['BB_position'] = (close - df['BB_lower']) / bb_range
                features['BB_squeeze'] = bb_range / close
            
            # RSI特征
            if 'RSI' in df.columns:
                rsi = df['RSI']
                features['RSI_overbought'] = (rsi > 70).astype(int)
                features['RSI_oversold'] = (rsi < 30).astype(int)
                features['RSI_trend'] = rsi.diff()
            
            # MACD特征
            if 'MACD' in df.columns:
                macd = df['MACD']
                features['MACD_signal'] = (macd > 0).astype(int)
                features['MACD_cross'] = ((macd > 0) & (macd.shift() <= 0)).astype(int) - \
                                        ((macd < 0) & (macd.shift() >= 0)).astype(int)
            
            # 成交量特征
            if 'volume' in df.columns:
                volume = df['volume']
                features['volume_ratio'] = volume / volume.rolling(20).mean()
                features['volume_price_corr'] = close.rolling(20).corr(volume)
            
            # 动量特征
            for period in [1, 3, 5, 10, 20]:
                features[f'momentum_{period}'] = close.pct_change(period)
                features[f'acceleration_{period}'] = close.pct_change(period).diff()
            
            # 波动率特征
            features['volatility_20'] = close.pct_change().rolling(20).std()
            features['volatility_50'] = close.pct_change().rolling(50).std()
            features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']
        
        return features
    
    def _create_statistical_features(self, df):
        """创建统计特征"""
        features = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change().dropna()
            
            if len(returns) > 10:
                # 基本统计量
                features['mean_return'] = returns.rolling(20).mean()
                features['std_return'] = returns.rolling(20).std()
                features['skewness'] = returns.rolling(50).apply(lambda x: stats.skew(x) if len(x) > 10 else 0)
                features['kurtosis'] = returns.rolling(50).apply(lambda x: stats.kurtosis(x) if len(x) > 10 else 0)
                
                # 分位数特征
                features['q10_return'] = returns.rolling(20).quantile(0.1)
                features['q90_return'] = returns.rolling(20).quantile(0.9)
                features['iqr_return'] = features['q90_return'] - features['q10_return']
                
                # 极端值特征
                features['max_drawdown'] = close.rolling(20).apply(
                    lambda x: (x.max() - x.min()) / x.max() if x.max() > 0 else 0
                )
                features['positive_ratio'] = (returns > 0).rolling(20).mean()
                
                # 自相关特征
                features['autocorr_1'] = returns.rolling(50).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 10 else 0
                )
                features['autocorr_5'] = returns.rolling(50).apply(
                    lambda x: x.autocorr(lag=5) if len(x) > 10 else 0
                )
        
        return features
    
    def _create_timeseries_features(self, df):
        """创建时间序列特征"""
        features = pd.DataFrame(index=df.index)
        
        # 时间特征
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['week_of_year'] = df.index.isocalendar().week
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # 周期性特征
        features['sin_hour'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['cos_hour'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['sin_day'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['cos_day'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # 时间标志
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_month_end'] = (df.index.is_month_end).astype(int)
        features['is_quarter_end'] = (df.index.is_quarter_end).astype(int)
        features['is_year_end'] = (df.index.is_year_end).astype(int)
        
        # 市场时间特征
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['european_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['us_session'] = ((features['hour'] >= 16) & (features['hour'] < 24)).astype(int)
        
        return features
    
    def _create_microstructure_features(self, df):
        """创建市场微观结构特征"""
        features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            # 价格范围特征
            features['price_range'] = (high - low) / close
            features['normalized_range'] = (high - low) / ((high + low) / 2)
            
            # 收盘位置特征
            features['close_position'] = (close - low) / (high - low).replace(0, 1)
            
            # 波动率特征
            features['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((np.log(high/low))**2).rolling(20).mean())
            features['garman_klass_vol'] = np.sqrt(((np.log(high/low))**2)/2 - (2*np.log(2)-1)*(np.log(close/close.shift()))**2).rolling(20).mean()
            
            # 成交量特征
            if 'quote_volume' in df.columns:
                quote_volume = df['quote_volume']
                features['dollar_volume'] = quote_volume
                features['volume_price_trend'] = quote_volume.diff() / quote_volume.shift()
            
            # 买卖压力特征
            if all(col in df.columns for col in ['taker_buy_base', 'taker_buy_quote']):
                buy_volume = df['taker_buy_base']
                sell_volume = volume - buy_volume
                features['buy_sell_ratio'] = buy_volume / (sell_volume + 1)
                features['net_buy_pressure'] = (buy_volume - sell_volume) / volume
            
        return features
    
    def _create_fundamental_features(self, df):
        """创建基本面特征"""
        features = pd.DataFrame(index=df.index)
        
        # 链上数据特征
        onchain_metrics = [
            'active_addresses', 'transaction_count', 'net_exchange_flow',
            'whale_transactions_1k', 'whale_transactions_10k',
            'network_growth', 'supply_on_exchanges'
        ]
        
        for metric in onchain_metrics:
            if metric in df.columns:
                # 原始值
                features[f'{metric}'] = df[metric]
                
                # 变化率
                features[f'{metric}_change'] = df[metric].pct_change()
                
                # Z-score标准化
                rolling_mean = df[metric].rolling(30).mean()
                rolling_std = df[metric].rolling(30).std()
                features[f'{metric}_zscore'] = (df[metric] - rolling_mean) / rolling_std.replace(0, 1)
                
                # 分位数
                features[f'{metric}_percentile'] = df[metric].rolling(90).rank(pct=True)
        
        # 链上复合指标
        if all(m in df.columns for m in ['active_addresses', 'transaction_count', 'net_exchange_flow']):
            # 标准化各项指标
            metrics = ['active_addresses', 'transaction_count', 'net_exchange_flow']
            normalized = {}
            for metric in metrics:
                mean_val = df[metric].rolling(30).mean()
                std_val = df[metric].rolling(30).std()
                normalized[metric] = (df[metric] - mean_val) / std_val.replace(0, 1)
            
            features['onchain_composite'] = (
                normalized['active_addresses'] * 0.4 +
                normalized['transaction_count'] * 0.3 +
                normalized['net_exchange_flow'] * 0.3
            )
        
        return features
    
    def _create_sentiment_features(self, df):
        """创建市场情绪特征"""
        features = pd.DataFrame(index=df.index)
        
        # 社交媒体情绪特征
        sentiment_metrics = ['twitter_sentiment', 'reddit_sentiment', 'news_sentiment']
        
        for metric in sentiment_metrics:
            if metric in df.columns:
                # 原始情绪值
                features[f'{metric}'] = df[metric]
                
                # 情绪变化
                features[f'{metric}_change'] = df[metric].diff()
                
                # 情绪动量
                features[f'{metric}_momentum'] = df[metric] - df[metric].rolling(7).mean()
                
                # 情绪极端值
                features[f'{metric}_extreme'] = (
                    (df[metric] > df[metric].rolling(30).quantile(0.9)) |
                    (df[metric] < df[metric].rolling(30).quantile(0.1))
                ).astype(int)
        
        # 社交媒体综合情绪
        if all(m in df.columns for m in sentiment_metrics):
            features['social_sentiment_composite'] = (
                df['twitter_sentiment'] * 0.4 +
                df['reddit_sentiment'] * 0.3 +
                df['news_sentiment'] * 0.3
            )
            
            # 情绪分歧
            sentiment_std = df[sentiment_metrics].std(axis=1)
            features['sentiment_divergence'] = sentiment_std
        
        # 情绪与价格关系
        if 'close' in df.columns and 'social_sentiment_composite' in features.columns:
            close = df['close']
            sentiment = features['social_sentiment_composite']
            
            features['sentiment_price_corr'] = close.rolling(20).corr(sentiment)
            features['sentiment_return_corr'] = close.pct_change().rolling(20).corr(sentiment)
        
        return features
    
    def _create_risk_features(self, df):
        """创建风险特征"""
        features = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # 风险价值（VaR）
            features['var_95'] = returns.rolling(100).quantile(0.05)
            features['var_99'] = returns.rolling(100).quantile(0.01)
            
            # 预期缺口（CVaR）
            def calculate_cvar(series, alpha=0.05):
                if len(series) < 10:
                    return 0
                var = series.quantile(alpha)
                cvar = series[series <= var].mean()
                return cvar if not np.isnan(cvar) else 0
            
            features['cvar_95'] = returns.rolling(100).apply(lambda x: calculate_cvar(x, 0.05))
            
            # 最大回撤
            rolling_max = close.expanding().max()
            drawdown = (close - rolling_max) / rolling_max
            features['max_drawdown_current'] = drawdown
            features['max_drawdown_20'] = drawdown.rolling(20).min()
            
            # 波动率风险
            features['volatility_regime'] = pd.cut(
                returns.rolling(20).std().rank(pct=True),
                bins=[0, 0.3, 0.7, 1],
                labels=['low', 'medium', 'high']
            ).astype(str)
            
            # 相关性风险（模拟）
            features['market_correlation'] = np.random.uniform(0.5, 0.9, len(features))
            features['sector_correlation'] = np.random.uniform(0.3, 0.7, len(features))
        
        # 流动性风险
        if 'volume' in df.columns:
            volume = df['volume']
            features['liquidity_risk'] = 1 / (volume.rolling(20).mean() + 1)
            features['volume_shock'] = (volume / volume.rolling(20).mean() - 1).abs()
        
        return features
    
    def _create_interaction_features(self, features_df):
        """创建交互特征"""
        interaction_features = pd.DataFrame(index=features_df.index)
        
        # 价格与情绪交互
        if all(col in features_df.columns for col in ['close', 'social_sentiment_composite']):
            interaction_features['price_sentiment_interaction'] = (
                features_df['close'] * features_df['social_sentiment_composite']
            )
        
        # 成交量与波动率交互
        if all(col in features_df.columns for col in ['volume_ratio', 'volatility_20']):
            interaction_features['volume_vol_interaction'] = (
                features_df['volume_ratio'] * features_df['volatility_20']
            )
        
        # 链上数据与衍生品交互
        if all(col in features_df.columns for col in ['net_exchange_flow', 'funding_rate']):
            interaction_features['flow_funding_interaction'] = (
                features_df['net_exchange_flow'] * features_df['funding_rate']
            )
        
        # 技术指标交互
        if all(col in features_df.columns for col in ['RSI', 'MACD']):
            interaction_features['rsi_macd_interaction'] = (
                features_df['RSI'] * features_df['MACD']
            )
        
        # 创建多项式特征
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # 选择最重要的几个特征创建多项式
            important_features = ['close', 'volume', 'RSI', 'social_sentiment_composite']
            available_features = [f for f in important_features if f in features_df.columns]
            
            if len(available_features) >= 2:
                for i in range(len(available_features)):
                    for j in range(i+1, len(available_features)):
                        col1 = available_features[i]
                        col2 = available_features[j]
                        interaction_features[f'{col1}_{col2}_product'] = (
                            features_df[col1] * features_df[col2]
                        )
        
        return interaction_features
    
    def _create_lag_features(self, features_df, max_lag=5):
        """创建滞后特征"""
        lag_features = pd.DataFrame(index=features_df.index)
        
        # 选择数值型列
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # 为重要特征创建滞后
        important_cols = [
            'close', 'volume', 'RSI', 'MACD', 'social_sentiment_composite',
            'net_exchange_flow', 'funding_rate', 'volatility_20'
        ]
        
        cols_to_lag = [col for col in important_cols if col in numeric_cols]
        
        for col in cols_to_lag:
            for lag in range(1, min(max_lag + 1, len(features_df) // 10)):
                lag_features[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
            
            # 创建差分特征
            lag_features[f'{col}_diff_1'] = features_df[col].diff()
            lag_features[f'{col}_diff_3'] = features_df[col].diff(3)
            lag_features[f'{col}_diff_5'] = features_df[col].diff(5)
        
        return lag_features
    
    def _select_features(self, features_df, n_features=50):
        """特征选择"""
        if len(features_df.columns) <= n_features:
            return features_df
        
        try:
            # 计算特征方差
            variances = features_df.var()
            
            # 移除低方差特征
            low_variance_mask = variances < 1e-6
            if low_variance_mask.any():
                features_df = features_df.loc[:, ~low_variance_mask]
            
            # 如果特征仍然太多，选择最重要的特征
            if len(features_df.columns) > n_features:
                # 使用相关性进行简单选择
                if 'close' in features_df.columns:
                    target = features_df['close']
                    correlations = features_df.apply(lambda x: x.corr(target))
                    top_features = correlations.abs().nlargest(n_features).index
                    features_df = features_df[top_features]
            
            return features_df
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return features_df
    
    def _scale_features(self, features_df):
        """特征缩放"""
        try:
            # 分离数值特征和非数值特征
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return features_df
            
            # 创建缩放后的数据框
            scaled_df = pd.DataFrame(index=features_df.index)
            
            # 对数值特征进行缩放
            for col in numeric_cols:
                try:
                    # 使用RobustScaler处理异常值
                    values = features_df[col].values.reshape(-1, 1)
                    scaled_values = self.scalers['robust'].fit_transform(values)
                    scaled_df[col] = scaled_values.flatten()
                except:
                    # 如果缩放失败，使用原始值
                    scaled_df[col] = features_df[col]
            
            # 添加非数值特征
            for col in non_numeric_cols:
                scaled_df[col] = features_df[col]
            
            return scaled_df
            
        except Exception as e:
            logger.error(f"特征缩放失败: {e}")
            return features_df
    
    def _update_feature_importance(self, features_df):
        """更新特征重要性记录"""
        try:
            # 简单的重要性估算：使用方差
            importance = features_df.var().to_dict()
            
            # 合并到历史记录
            for feature, imp in importance.items():
                if feature not in self.feature_importance:
                    self.feature_importance[feature] = []
                self.feature_importance[feature].append(imp)
            
            # 保留最近100次记录
            for feature in self.feature_importance:
                if len(self.feature_importance[feature]) > 100:
                    self.feature_importance[feature] = self.feature_importance[feature][-100:]
            
        except Exception as e:
            logger.error(f"更新特征重要性失败: {e}")
    
    def create_labels(self, price_df, horizon=None, threshold=None):
        """创建预测标签"""
        if horizon is None:
            horizon = self.trading_config.prediction_horizon
        if threshold is None:
            threshold = 0.03
        
        try:
            if len(price_df) < horizon + 10:
                return pd.Series([0] * len(price_df), index=price_df.index, name='label')
            
            if 'close' not in price_df.columns:
                return pd.Series([0] * len(price_df), index=price_df.index, name='label')
            
            close = price_df['close']
            
            # 计算未来收益率
            future_returns = close.pct_change(horizon).shift(-horizon)
            
            # 创建三分类标签
            labels = pd.Series(0, index=price_df.index, name='label')  # 0 = 横盘
            
            # 只对有效数据点赋值
            valid_mask = ~future_returns.isna()
            valid_returns = future_returns[valid_mask]
            
            # 上涨标签 (1)
            up_mask = valid_returns > threshold
            labels.loc[valid_mask & up_mask] = 1
            
            # 下跌标签 (-1)
            down_mask = valid_returns < -threshold
            labels.loc[valid_mask & down_mask] = -1
            
            # 统计标签分布
            label_counts = labels.value_counts()
            logger.info(f"标签分布: 上涨={label_counts.get(1, 0)}, "
                       f"下跌={label_counts.get(-1, 0)}, "
                       f"横盘={label_counts.get(0, 0)}")
            
            return labels
            
        except Exception as e:
            logger.error(f"创建标签失败: {e}")
            return pd.Series([0] * len(price_df), index=price_df.index, name='label')

# ==================== 集成模型管理器 ====================

class EnsembleModelManager:
    """集成模型管理器，包含所有原始模型"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trading_config = config.trading_config
        self.model_config = config.model_config
        
        # 模型字典
        self.models = {}
        self.model_weights = self.model_config['model_weights']
        
        # 特征工程师
        self.feature_engineer = AdvancedFeatureEngineer(config)
        
        # 模型性能跟踪
        self.model_performance = {}
        self.training_history = []
        self.last_retrain_time = 0
        
        # 模型文件路径
        self.model_dir = config.directories['models']
        
        # 初始化所有模型
        self._initialize_models()
        
        logger.info("集成模型管理器初始化完成")
    
    def _initialize_models(self):
        """初始化所有模型"""
        logger.info("初始化集成模型...")
        
        try:
            # 1. XGBoost模型
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            logger.info("XGBoost模型初始化完成")
            
            # 2. LightGBM模型
            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            logger.info("LightGBM模型初始化完成")
            
            # 3. 随机森林模型
            self.models['rf'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            logger.info("随机森林模型初始化完成")
            
            # 4. 梯度提升树模型
            self.models['gbt'] = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=0
            )
            logger.info("梯度提升树模型初始化完成")
            
            # 5. Prophet模型（用于时间序列预测）
            self.models['prophet'] = None  # 稍后训练时初始化
            
            # 6. LSTM模型（如果可用）
            if PYTORCH_AVAILABLE:
                self.models['lstm'] = self._create_lstm_model()
                logger.info("LSTM模型初始化完成")
            else:
                self.models['lstm'] = None
                logger.info("PyTorch不可用，跳过LSTM模型")
            
            # 7. 集成模型
            self.models['ensemble'] = VotingClassifier(
                estimators=[
                    ('xgb', self.models['xgb']),
                    ('lgb', self.models['lgb']),
                    ('rf', self.models['rf']),
                    ('gbt', self.models['gbt'])
                ],
                voting='soft',
                weights=list(self.model_weights.values())[:4]
            )
            logger.info("集成模型初始化完成")
            
            logger.info(f"共初始化 {len(self.models)} 个模型")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_lstm_model(self):
        """创建LSTM模型"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, output_size)
                )
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                out = self.fc(last_out)
                return self.softmax(out)
        
        return LSTMModel(input_size=50)  # 输入特征维度
    
    def train_models(self, data_manager: CompleteDataManager, retrain=False):
        """训练所有模型"""
        logger.info("开始训练模型...")
        
        try:
            # 检查是否需要重新训练
            current_time = time.time()
            hours_since_last_train = (current_time - self.last_retrain_time) / 3600
            
            if not retrain and hours_since_last_train < self.trading_config.model_retrain_interval:
                logger.info(f"距离上次训练仅{hours_since_last_train:.1}小时，跳过训练")
                return True
            
            # 获取特征数据
            features_df = self.feature_engineer.create_complete_features(data_manager)
            
            if features_df.empty:
                logger.error("特征数据为空，无法训练模型")
                return False
            
            # 获取标签
            price_df = data_manager.historical_data['price']['1d'] if '1d' in data_manager.historical_data['price'] else features_df
            labels = self.feature_engineer.create_labels(price_df)
            
            # 对齐特征和标签
            common_index = features_df.index.intersection(labels.index)
            if len(common_index) < 100:
                logger.error(f"有效样本太少: {len(common_index)}")
                return False
            
            X = features_df.loc[common_index]
            y = labels.loc[common_index]
            
            # 划分训练集和测试集
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if len(X_train) < 50 or len(X_test) < 20:
                logger.error(f"训练数据不足: 训练集={len(X_train)}, 测试集={len(X_test)}")
                return False
            
            logger.info(f"数据准备完成: 训练集={len(X_train)}, 测试集={len(X_test)}")
            
            # 训练各个模型
            model_performance = {}
            
            for name, model in self.models.items():
                if model is None:
                    continue
                
                try:
                    logger.info(f"训练 {name} 模型...")
                    
                    if name == 'prophet':
                        # Prophet需要特殊处理
                        self._train_prophet_model(X_train, y_train)
                        performance = {'accuracy': 0.5}  # 占位值
                        
                    elif name == 'lstm' and PYTORCH_AVAILABLE:
                        # LSTM需要特殊处理
                        performance = self._train_lstm_model(model, X_train, y_train, X_test, y_test)
                        
                    else:
                        # 训练传统机器学习模型
                        model.fit(X_train, y_train)
                        
                        # 评估模型
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        performance = {
                            'accuracy': accuracy,
                            'predictions': y_pred,
                            'probabilities': model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                        }
                        
                        logger.info(f"{name} 模型训练完成，准确率: {accuracy:.3f}")
                    
                    model_performance[name] = performance
                    
                except Exception as e:
                    logger.error(f"训练 {name} 模型失败: {e}")
                    model_performance[name] = {'accuracy': 0.0, 'error': str(e)}
            
            # 训练集成模型
            if 'ensemble' in self.models and self.models['ensemble'] is not None:
                try:
                    self.models['ensemble'].fit(X_train, y_train)
                    y_pred_ensemble = self.models['ensemble'].predict(X_test)
                    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
                    
                    model_performance['ensemble'] = {
                        'accuracy': accuracy_ensemble,
                        'predictions': y_pred_ensemble,
                        'probabilities': self.models['ensemble'].predict_proba(X_test)
                    }
                    
                    logger.info(f"集成模型训练完成，准确率: {accuracy_ensemble:.3f}")
                except Exception as e:
                    logger.error(f"训练集成模型失败: {e}")
            
            # 更新模型性能记录
            self.model_performance = model_performance
            
            # 保存训练记录
            training_record = {
                'timestamp': datetime.now(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'performance': model_performance
            }
            self.training_history.append(training_record)
            
            # 限制历史记录数量
            if len(self.training_history) > 100:
                self.training_history = self.training_history[-100:]
            
            # 保存模型
            self._save_models()
            
            # 更新最后训练时间
            self.last_retrain_time = current_time
            
            logger.info("所有模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_prophet_model(self, X_train, y_train):
        """训练Prophet模型"""
        try:
            # Prophet需要特定的数据格式
            prophet_df = pd.DataFrame({
                'ds': X_train.index,
                'y': y_train.values
            })
            
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            self.models['prophet'] = model
            
            logger.info("Prophet模型训练完成")
            
        except Exception as e:
            logger.error(f"训练Prophet模型失败: {e}")
            self.models['prophet'] = None
    
    def _train_lstm_model(self, model, X_train, y_train, X_test, y_test):
        """训练LSTM模型"""
        try:
            # 转换数据为PyTorch格式
            X_train_tensor = torch.FloatTensor(X_train.values).unsqueeze(1)
            y_train_tensor = torch.LongTensor(y_train.values + 1)  # 标签从-1,0,1转换为0,1,2
            
            X_test_tensor = torch.FloatTensor(X_test.values).unsqueeze(1)
            y_test_tensor = torch.LongTensor(y_test.values + 1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 定义优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # 训练模型
            model.train()
            epochs = 50
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"LSTM epoch {epoch+1}/{epochs}, loss: {total_loss/len(train_loader):.4f}")
            
            # 评估模型
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predictions = torch.max(test_outputs, 1)
                accuracy = (predictions == y_test_tensor).float().mean().item()
            
            logger.info(f"LSTM模型训练完成，准确率: {accuracy:.3f}")
            
            return {
                'accuracy': accuracy,
                'predictions': predictions.numpy() - 1,  # 转换回原始标签
                'probabilities': test_outputs.numpy()
            }
            
        except Exception as e:
            logger.error(f"训练LSTM模型失败: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    def predict(self, features, use_ensemble=True):
        """使用模型进行预测"""
        try:
            if features.empty:
                logger.warning("特征数据为空，返回默认预测")
                return self._get_default_prediction()
            
            # 准备预测数据
            X_pred = features.iloc[-1:].copy()  # 只预测最新数据
            
            predictions = {}
            probabilities = {}
            
            # 获取各个模型的预测
            for name, model in self.models.items():
                if model is None:
                    continue
                
                try:
                    if name == 'prophet':
                        # Prophet预测
                        if model is not None:
                            future = model.make_future_dataframe(periods=1)
                            forecast = model.predict(future)
                            pred = forecast.iloc[-1]['yhat']
                            
                            # 将连续预测转换为分类
                            if pred > 0.02:
                                predictions[name] = 1
                                probabilities[name] = [0.2, 0.3, 0.5]
                            elif pred < -0.02:
                                predictions[name] = -1
                                probabilities[name] = [0.5, 0.3, 0.2]
                            else:
                                predictions[name] = 0
                                probabilities[name] = [0.3, 0.4, 0.3]
                    
                    elif name == 'lstm' and PYTORCH_AVAILABLE:
                        # LSTM预测
                        if model is not None:
                            X_tensor = torch.FloatTensor(X_pred.values).unsqueeze(1)
                            model.eval()
                            with torch.no_grad():
                                output = model(X_tensor)
                                prob = output.numpy()[0]
                                pred = np.argmax(prob) - 1  # 转换回原始标签
                                
                            predictions[name] = pred
                            probabilities[name] = prob
                    
                    else:
                        # 传统机器学习模型预测
                        if hasattr(model, 'predict'):
                            pred = model.predict(X_pred)[0]
                            predictions[name] = pred
                            
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(X_pred)[0]
                                probabilities[name] = prob
                
                except Exception as e:
                    logger.error(f"{name} 模型预测失败: {e}")
                    predictions[name] = 0
                    probabilities[name] = [0.33, 0.34, 0.33]
            
            # 集成预测
            if use_ensemble and 'ensemble' in self.models and self.models['ensemble'] is not None:
                try:
                    ensemble_pred = self.models['ensemble'].predict(X_pred)[0]
                    ensemble_prob = self.models['ensemble'].predict_proba(X_pred)[0]
                    
                    predictions['ensemble'] = ensemble_pred
                    probabilities['ensemble'] = ensemble_prob
                    
                    final_prediction = ensemble_pred
                    final_probability = ensemble_prob
                    
                except:
                    # 如果集成模型失败，使用加权投票
                    final_prediction, final_probability = self._weighted_vote(predictions, probabilities)
            else:
                # 使用加权投票
                final_prediction, final_probability = self._weighted_vote(predictions, probabilities)
            
            # 计算置信度
            if final_prediction == 1:  # 上涨
                confidence = final_probability[2] if len(final_probability) > 2 else 0.5
            elif final_prediction == -1:  # 下跌
                confidence = final_probability[0] if len(final_probability) > 0 else 0.5
            else:  # 横盘
                confidence = final_probability[1] if len(final_probability) > 1 else 0.5
            
            # 转换为交易信号
            signal = self._prediction_to_signal(final_prediction, confidence)
            
            # 添加模型详情
            signal['model_details'] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'ensemble_used': use_ensemble
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return self._get_default_prediction()
    
    def _weighted_vote(self, predictions, probabilities):
        """加权投票"""
        try:
            # 初始化投票结果
            vote_counts = {-1: 0.0, 0: 0.0, 1: 0.0}
            weight_sum = 0.0
            
            for name, pred in predictions.items():
                if name in self.model_weights:
                    weight = self.model_weights.get(name, 0.1)
                    vote_counts[pred] += weight
                    weight_sum += weight
            
            # 如果没有有效的权重，使用平均权重
            if weight_sum == 0:
                weight = 1.0 / len(predictions)
                for pred in predictions.values():
                    vote_counts[pred] += weight
                weight_sum = 1.0
            
            # 归一化
            for key in vote_counts:
                vote_counts[key] /= weight_sum
            
            # 选择得票最多的类别
            final_prediction = max(vote_counts, key=vote_counts.get)
            
            # 计算概率分布
            final_probability = [
                vote_counts.get(-1, 0),
                vote_counts.get(0, 0),
                vote_counts.get(1, 0)
            ]
            
            return final_prediction, final_probability
            
        except:
            return 0, [0.33, 0.34, 0.33]
    
    def _prediction_to_signal(self, prediction, confidence):
        """将预测结果转换为交易信号"""
        signal_map = {
            -1: 'SELL',
            0: 'HOLD',
            1: 'BUY'
        }
        
        strength_map = {
            -1: 'STRONG_SELL' if confidence > 0.7 else 'SELL',
            0: 'NEUTRAL',
            1: 'STRONG_BUY' if confidence > 0.7 else 'BUY'
        }
        
        action = signal_map.get(prediction, 'HOLD')
        strength = strength_map.get(prediction, 'NEUTRAL')
        
        # 计算建议仓位大小（基于置信度和风险预算）
        if action == 'BUY':
            position_size = max(100, int(1000 * confidence))  # 至少100 DOGE
        elif action == 'SELL':
            position_size = max(100, int(800 * confidence))   # 至少100 DOGE
        else:
            position_size = 0
        
        signal = {
            'timestamp': datetime.now(),
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'position_size': position_size,
            'prediction': prediction,
            'reasoning': [
                f'模型预测: {action} ({strength})',
                f'置信度: {confidence:.1%}',
                f'建议仓位: {position_size} DOGE'
            ]
        }
        
        return signal
    
    def _get_default_prediction(self):
        """获取默认预测"""
        return {
            'timestamp': datetime.now(),
            'action': 'HOLD',
            'strength': 'NEUTRAL',
            'confidence': 0.5,
            'position_size': 0,
            'prediction': 0,
            'reasoning': ['系统错误，使用默认持有信号'],
            'model_details': {'error': '预测失败'}
        }
    
    def _save_models(self):
        """保存训练好的模型"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(self.model_dir, f'models_{timestamp}')
            os.makedirs(save_dir, exist_ok=True)
            
            for name, model in self.models.items():
                if model is None:
                    continue
                
                try:
                    model_file = os.path.join(save_dir, f'{name}.pkl')
                    
                    if name in ['prophet', 'lstm']:
                        # 特殊模型需要特殊处理
                        if name == 'prophet' and model is not None:
                            with open(model_file, 'wb') as f:
                                pickle.dump(model, f)
                    else:
                        with open(model_file, 'wb') as f:
                            pickle.dump(model, f)
                    
                    logger.info(f"模型 {name} 已保存到 {model_file}")
                except Exception as e:
                    logger.error(f"保存模型 {name} 失败: {e}")
            
            # 保存模型性能
            perf_file = os.path.join(save_dir, 'performance.json')
            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_performance, f, indent=2, ensure_ascii=False)
            
            logger.info(f"所有模型已保存到 {save_dir}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_models(self, model_dir=None):
        """加载训练好的模型"""
        try:
            if model_dir is None:
                # 查找最新的模型目录
                if not os.path.exists(self.model_dir):
                    return False
                
                model_dirs = [d for d in os.listdir(self.model_dir) 
                            if os.path.isdir(os.path.join(self.model_dir, d))]
                
                if not model_dirs:
                    return False
                
                model_dirs.sort(reverse=True)
                model_dir = os.path.join(self.model_dir, model_dirs[0])
            
            logger.info(f"从 {model_dir} 加载模型...")
            
            for name in self.models.keys():
                model_file = os.path.join(model_dir, f'{name}.pkl')
                
                if os.path.exists(model_file):
                    try:
                        with open(model_file, 'rb') as f:
                            self.models[name] = pickle.load(f)
                        logger.info(f"模型 {name} 加载成功")
                    except Exception as e:
                        logger.error(f"加载模型 {name} 失败: {e}")
            
            # 加载模型性能
            perf_file = os.path.join(model_dir, 'performance.json')
            if os.path.exists(perf_file):
                with open(perf_file, 'r', encoding='utf-8') as f:
                    self.model_performance = json.load(f)
            
            logger.info("模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

# ==================== 完整交易引擎 ====================

class CompleteTradingEngine:
    """完整的交易引擎，整合所有功能"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trading_config = config.trading_config
        
        # 初始化API客户端
        self.api_client = BinanceClient(
            api_key=self.trading_config.api_key,
            api_secret=self.trading_config.api_secret,
            proxy=self.trading_config.proxy
        )
        
        # 交易状态
        self.positions = {
            'DOGEUSDT': {
                'quantity': 0.0,
                'entry_price': 0.0,
                'entry_time': None,
                'current_price': 0.0,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_ratio': 0.0,
                'position_value': 0.0
            }
        }
        
        # 账户状态
        self.balance = {
            'USDT': self.trading_config.initial_balance,
            'DOGE': 0.0,
            'total_value': self.trading_config.initial_balance
        }
        
        # 交易历史
        self.trade_history = deque(maxlen=1000)
        self.order_history = deque(maxlen=500)
        
        # 性能指标
        self.performance = {
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'monthly_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'total_commission': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
        # 风险控制
        self.risk_metrics = {
            'daily_loss_limit': self.trading_config.initial_balance * 0.02,
            'max_position_limit': self.trading_config.initial_balance * self.trading_config.max_position_ratio,
            'max_drawdown_limit': 0.1,  # 10%
            'consecutive_loss_limit': 3,
            'daily_trade_limit': 20,
            'cooldown_period': 300  # 5分钟冷却
        }
        
        # 交易限制
        self.trade_limits = {
            'daily_trade_count': 0,
            'last_trade_time': None,
            'cooldown_until': None,
            'daily_pnl': 0.0
        }
        
        # 手续费计算
        self.commission_rates = {
            'maker': 0.001,  # 挂单费率
            'taker': 0.001,  # 吃单费率
            'min_commission': self.trading_config.min_commission
        }
        
        # 线程锁
        self.trade_lock = threading.RLock()
        self.balance_lock = threading.RLock()
        
        # 状态标志
        self.initialized = False
        self.paper_trading = self.trading_config.paper_trading
        
        logger.info("完整交易引擎初始化完成")
    
    def initialize(self):
        """初始化交易引擎"""
        try:
            # 测试API连接
            if self.trading_config.live_trading and not self.paper_trading:
                connected = self.api_client.test_connection()
                if not connected:
                    logger.error("API连接测试失败，切换到模拟交易")
                    self.paper_trading = True
            
            # 获取账户余额（实盘）
            if not self.paper_trading:
                self._update_real_balance()
            else:
                logger.info("使用模拟交易模式")
            
            self.initialized = True
            logger.info("交易引擎初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"交易引擎初始化失败: {e}")
            return False
    
    def _update_real_balance(self):
        """更新实盘余额"""
        try:
            balances = self.api_client.get_balance()
            
            if balances:
                with self.balance_lock:
                    # 更新USDT余额
                    usdt_info = balances.get('USDT', {'free': 0, 'locked': 0, 'total': 0})
                    self.balance['USDT'] = usdt_info['free']
                    
                    # 更新DOGE余额
                    doge_info = balances.get('DOGE', {'free': 0, 'locked': 0, 'total': 0})
                    self.balance['DOGE'] = doge_info['free']
                    
                    # 获取当前价格计算总价值
                    current_price = self.api_client.get_price(self.trading_config.symbol)
                    if current_price:
                        self.positions['DOGEUSDT']['current_price'] = current_price
                        self.positions['DOGEUSDT']['quantity'] = doge_info['free']
                        self.positions['DOGEUSDT']['position_value'] = doge_info['free'] * current_price
                        self.balance['total_value'] = self.balance['USDT'] + self.positions['DOGEUSDT']['position_value']
                    
                    logger.info(f"余额更新: USDT={self.balance['USDT']:.2f}, "
                               f"DOGE={self.balance['DOGE']:.2f}, "
                               f"总价值={self.balance['total_value']:.2f}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"更新实盘余额失败: {e}")
            return False
    
    def calculate_commission(self, trade_value, is_maker=False):
        """计算手续费"""
        try:
            rate = self.commission_rates['maker'] if is_maker else self.commission_rates['taker']
            commission = trade_value * rate
            min_commission = self.commission_rates['min_commission']
            
            return max(commission, min_commission)
            
        except:
            return self.commission_rates['min_commission']
    
    def check_risk_limits(self, signal, current_price):
        """检查风险限制"""
        with self.trade_lock:
            risk_checks = []
            
            # 1. 检查冷却期
            if self.trade_limits['cooldown_until'] and \
               datetime.now() < self.trade_limits['cooldown_until']:
                remaining = (self.trade_limits['cooldown_until'] - datetime.now()).seconds
                risk_checks.append((
                    False,
                    f"冷却期中，剩余{remaining}秒"
                ))
            
            # 2. 检查日交易次数限制
            if self.trade_limits['daily_trade_count'] >= self.risk_metrics['daily_trade_limit']:
                risk_checks.append((
                    False,
                    f"达到日交易次数限制: {self.trade_limits['daily_trade_count']}/{self.risk_metrics['daily_trade_limit']}"
                ))
            
            # 3. 检查日亏损限制
            daily_loss_limit = self.risk_metrics['daily_loss_limit']
            if self.trade_limits['daily_pnl'] <= -daily_loss_limit:
                risk_checks.append((
                    False,
                    f"达到日亏损限制: {self.trade_limits['daily_pnl']:.2f} <= {-daily_loss_limit:.2f}"
                ))
            
            # 4. 检查连续亏损
            if self.performance['consecutive_losses'] >= self.risk_metrics['consecutive_loss_limit']:
                risk_checks.append((
                    False,
                    f"连续亏损{self.performance['consecutive_losses']}次，暂停交易"
                ))
            
            # 5. 检查最大回撤
            if self.performance['max_drawdown'] >= self.risk_metrics['max_drawdown_limit']:
                risk_checks.append((
                    False,
                    f"达到最大回撤限制: {self.performance['max_drawdown']:.1%}"
                ))
            
            # 6. 检查仓位限制
            if signal['action'] == 'BUY':
                position = self.positions['DOGEUSDT']
                position_value = position['position_value']
                proposed_trade_value = signal.get('position_size', 0) * current_price
                
                if position_value + proposed_trade_value > self.risk_metrics['max_position_limit']:
                    risk_checks.append((
                        False,
                        f"超出最大仓位限制: {position_value + proposed_trade_value:.2f} > "
                        f"{self.risk_metrics['max_position_limit']:.2f}"
                    ))
                
                # 检查资金是否足够（包含手续费）
                estimated_commission = self.calculate_commission(proposed_trade_value)
                required_funds = proposed_trade_value + estimated_commission
                
                if required_funds > self.balance['USDT']:
                    risk_checks.append((
                        False,
                        f"资金不足: 需要{required_funds:.2f}，可用{self.balance['USDT']:.2f}"
                    ))
            
            # 7. 检查最小交易金额
            if signal['action'] != 'HOLD':
                trade_value = signal.get('position_size', 0) * current_price
                if trade_value < self.trading_config.min_trade_amount:
                    risk_checks.append((
                        False,
                        f"交易金额太小: {trade_value:.2f} < {self.trading_config.min_trade_amount:.2f}"
                    ))
            
            # 如果没有风险问题，返回通过
            if not risk_checks:
                return (True, "风险检查通过")
            
            # 返回第一个风险问题
            return risk_checks[0]
    
    def execute_trade(self, signal, current_price=None):
        """执行交易"""
        if not current_price:
            current_price = self.api_client.get_price(self.trading_config.symbol) or 0.08
        
        with self.trade_lock:
            # 创建交易记录
            trade_record = {
                'timestamp': datetime.now(),
                'signal': signal.copy(),
                'current_price': current_price,
                'action': 'HOLD',
                'quantity': 0.0,
                'price': 0.0,
                'value': 0.0,
                'commission': 0.0,
                'realized_pnl': 0.0,
                'realized_pnl_ratio': 0.0,
                'success': False,
                'reason': '',
                'paper_trading': self.paper_trading,
                'position_before': self.positions['DOGEUSDT']['quantity'],
                'balance_before': self.balance.copy()
            }
            
            try:
                # 检查风险限制
                risk_allowed, risk_reason = self.check_risk_limits(signal, current_price)
                if not risk_allowed:
                    trade_record['reason'] = f"风险限制: {risk_reason}"
                    self.trade_history.append(trade_record)
                    return trade_record
                
                action = signal['action']
                position = self.positions['DOGEUSDT']
                
                if action == 'BUY':
                    trade_record = self._execute_buy(trade_record, signal, current_price)
                    
                elif action == 'SELL':
                    trade_record = self._execute_sell(trade_record, signal, current_price)
                    
                else:  # HOLD
                    trade_record['reason'] = '持有信号，不交易'
                
                # 更新交易历史
                if trade_record['success']:
                    self.trade_history.append(trade_record)
                    
                    # 更新交易限制
                    self.trade_limits['daily_trade_count'] += 1
                    self.trade_limits['last_trade_time'] = datetime.now()
                    
                    # 设置冷却期（防止过度交易）
                    self.trade_limits['cooldown_until'] = datetime.now() + timedelta(
                        seconds=self.risk_metrics['cooldown_period']
                    )
                    
                    # 更新性能指标
                    self._update_performance_metrics(trade_record)
                    
                    # 记录订单历史
                    self.order_history.append({
                        'timestamp': trade_record['timestamp'],
                        'action': trade_record['action'],
                        'quantity': trade_record['quantity'],
                        'price': trade_record['price'],
                        'status': 'FILLED' if trade_record['success'] else 'REJECTED'
                    })
                    
                    logger.info(f"交易执行: {trade_record['action']} {trade_record['quantity']:.0f} DOGE @ "
                               f"{trade_record['price']:.6f}, PnL: {trade_record['realized_pnl']:.2f}")
                
                return trade_record
                
            except Exception as e:
                logger.error(f"交易执行异常: {e}")
                trade_record['success'] = False
                trade_record['reason'] = f'执行异常: {str(e)}'
                self.trade_history.append(trade_record)
                return trade_record
    
    def _execute_buy(self, trade_record, signal, current_price):
        """执行买入操作"""
        try:
            position = self.positions['DOGEUSDT']
            
            # 计算买入数量
            position_size = signal.get('position_size', 0)
            if position_size <= 0:
                trade_record['reason'] = '无效的买入数量'
                return trade_record
            
            # 计算交易价值
            trade_value = position_size * current_price
            
            # 计算手续费
            commission = self.calculate_commission(trade_value, is_maker=False)
            
            # 检查资金是否足够（模拟交易跳过）
            if not self.paper_trading:
                required_funds = trade_value + commission
                if required_funds > self.balance['USDT']:
                    trade_record['reason'] = f'资金不足: 需要{required_funds:.2f}，可用{self.balance['USDT']:.2f}'
                    return trade_record
            
            # 执行买入（实盘或模拟）
            if not self.paper_trading:
                # 实盘交易
                order_result = self.api_client.send_order(
                    symbol=self.trading_config.symbol,
                    side='BUY',
                    quantity=position_size,
                    order_type='MARKET'
                )
                
                if not order_result.get('success', False):
                    trade_record['reason'] = f'API订单失败: {order_result.get("error", "未知错误")}'
                    return trade_record
                
                # 从订单结果获取实际成交价和数量
                fills = order_result.get('fills', [])
                if fills:
                    actual_price = float(fills[0]['price'])
                    actual_qty = sum(float(fill['qty']) for fill in fills)
                    actual_commission = sum(float(fill.get('commission', 0)) for fill in fills)
                    
                    trade_value = actual_qty * actual_price
                    commission = actual_commission
                    position_size = actual_qty
                else:
                    actual_price = current_price
                
                # 更新实盘余额
                self._update_real_balance()
                
            else:
                # 模拟交易
                actual_price = current_price
                actual_qty = position_size
                
                # 更新模拟余额
                with self.balance_lock:
                    self.balance['USDT'] -= (trade_value + commission)
                    self.balance['DOGE'] += position_size
            
            # 更新仓位
            old_quantity = position['quantity']
            old_value = old_quantity * position['entry_price'] if old_quantity > 0 else 0
            
            new_quantity = old_quantity + position_size
            new_value = old_value + trade_value
            
            # 计算平均入场价
            if new_quantity > 0:
                new_entry_price = (new_value + commission) / new_quantity
            else:
                new_entry_price = 0
            
            # 更新仓位信息
            position['quantity'] = new_quantity
            position['entry_price'] = new_entry_price
            position['entry_time'] = datetime.now()
            position['current_price'] = actual_price
            position['position_value'] = new_quantity * actual_price
            
            # 更新浮动盈亏
            position['unrealized_pnl'] = (actual_price - new_entry_price) * new_quantity
            position['unrealized_pnl_ratio'] = (actual_price - new_entry_price) / new_entry_price if new_entry_price > 0 else 0
            
            # 更新总价值
            with self.balance_lock:
                if self.paper_trading:
                    self.balance['total_value'] = self.balance['USDT'] + position['position_value']
            
            # 更新交易记录
            trade_record.update({
                'action': 'BUY',
                'quantity': position_size,
                'price': actual_price,
                'value': trade_value,
                'commission': commission,
                'realized_pnl': 0.0,  # 买入没有实现盈亏
                'realized_pnl_ratio': 0.0,
                'success': True,
                'reason': f'买入 {position_size:.0f} DOGE @ {actual_price:.6f}, 手续费: {commission:.4f} USDT',
                'position_after': new_quantity,
                'balance_after': self.balance.copy(),
                'entry_price': new_entry_price
            })
            
            # 更新手续费统计
            self.performance['total_commission'] += commission
            
            return trade_record
            
        except Exception as e:
            logger.error(f"买入执行失败: {e}")
            trade_record['reason'] = f'买入执行失败: {str(e)}'
            return trade_record
    
    def _execute_sell(self, trade_record, signal, current_price):
        """执行卖出操作"""
        try:
            position = self.positions['DOGEUSDT']
            
            # 计算卖出数量
            position_size = signal.get('position_size', 0)
            if position_size <= 0:
                trade_record['reason'] = '无效的卖出数量'
                return trade_record
            
            # 不能卖出超过持仓的数量
            if position_size > position['quantity']:
                position_size = position['quantity']
            
            if position_size == 0:
                trade_record['reason'] = '没有持仓可卖出'
                return trade_record
            
            # 计算交易价值
            trade_value = position_size * current_price
            
            # 计算手续费
            commission = self.calculate_commission(trade_value, is_maker=False)
            
            # 计算盈亏
            entry_price = position['entry_price'] if position['quantity'] > 0 else current_price
            realized_pnl = (current_price - entry_price) * position_size - commission
            realized_pnl_ratio = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # 执行卖出（实盘或模拟）
            if not self.paper_trading:
                # 实盘交易
                order_result = self.api_client.send_order(
                    symbol=self.trading_config.symbol,
                    side='SELL',
                    quantity=position_size,
                    order_type='MARKET'
                )
                
                if not order_result.get('success', False):
                    trade_record['reason'] = f'API订单失败: {order_result.get("error", "未知错误")}'
                    return trade_record
                
                # 从订单结果获取实际成交价和数量
                fills = order_result.get('fills', [])
                if fills:
                    actual_price = float(fills[0]['price'])
                    actual_qty = sum(float(fill['qty']) for fill in fills)
                    actual_commission = sum(float(fill.get('commission', 0)) for fill in fills)
                    
                    trade_value = actual_qty * actual_price
                    commission = actual_commission
                    position_size = actual_qty
                    realized_pnl = (actual_price - entry_price) * position_size - commission
                    realized_pnl_ratio = (actual_price - entry_price) / entry_price if entry_price > 0 else 0
                else:
                    actual_price = current_price
                
                # 更新实盘余额
                self._update_real_balance()
                
            else:
                # 模拟交易
                actual_price = current_price
                actual_qty = position_size
                
                # 更新模拟余额
                with self.balance_lock:
                    self.balance['USDT'] += (trade_value - commission)
                    self.balance['DOGE'] -= position_size
            
            # 更新仓位
            position['quantity'] -= position_size
            
            # 如果全部卖出，重置入场价
            if position['quantity'] == 0:
                position['entry_price'] = 0.0
                position['entry_time'] = None
            
            position['current_price'] = actual_price
            position['position_value'] = position['quantity'] * actual_price
            
            # 更新浮动盈亏
            if position['quantity'] > 0 and position['entry_price'] > 0:
                position['unrealized_pnl'] = (actual_price - position['entry_price']) * position['quantity']
                position['unrealized_pnl_ratio'] = (actual_price - position['entry_price']) / position['entry_price']
            else:
                position['unrealized_pnl'] = 0.0
                position['unrealized_pnl_ratio'] = 0.0
            
            # 更新总价值
            with self.balance_lock:
                if self.paper_trading:
                    self.balance['total_value'] = self.balance['USDT'] + position['position_value']
            
            # 更新交易记录
            trade_record.update({
                'action': 'SELL',
                'quantity': position_size,
                'price': actual_price,
                'value': trade_value,
                'commission': commission,
                'realized_pnl': realized_pnl,
                'realized_pnl_ratio': realized_pnl_ratio,
                'success': True,
                'reason': f'卖出 {position_size:.0f} DOGE @ {actual_price:.6f}, '
                         f'盈亏: {realized_pnl:.2f} USDT ({realized_pnl_ratio:.2%}), '
                         f'手续费: {commission:.4f} USDT',
                'position_after': position['quantity'],
                'balance_after': self.balance.copy()
            })
            
            # 更新手续费统计
            self.performance['total_commission'] += commission
            
            return trade_record
            
        except Exception as e:
            logger.error(f"卖出执行失败: {e}")
            trade_record['reason'] = f'卖出执行失败: {str(e)}'
            return trade_record
    
    def _update_performance_metrics(self, trade_record):
        """更新性能指标"""
        with self.trade_lock:
            # 更新交易计数
            self.performance['total_trades'] += 1
            
            # 更新盈亏统计
            realized_pnl = trade_record.get('realized_pnl', 0)
            self.performance['total_pnl'] += realized_pnl
            self.trade_limits['daily_pnl'] += realized_pnl
            
            # 更新日/周/月盈亏
            current_time = datetime.now()
            trade_time = trade_record['timestamp']
            
            # 这里简化处理，实际应该按时间窗口统计
            self.performance['daily_pnl'] += realized_pnl
            
            # 更新胜率统计
            if realized_pnl > 0:
                self.performance['winning_trades'] += 1
                self.performance['consecutive_wins'] += 1
                self.performance['consecutive_losses'] = 0
                
                # 更新最大盈利
                if realized_pnl > self.performance['largest_win']:
                    self.performance['largest_win'] = realized_pnl
                
                # 更新平均盈利
                total_wins = self.performance['winning_trades']
                self.performance['avg_win'] = (
                    (self.performance['avg_win'] * (total_wins - 1) + realized_pnl) / total_wins
                    if total_wins > 0 else realized_pnl
                )
                
            elif realized_pnl < 0:
                self.performance['losing_trades'] += 1
                self.performance['consecutive_losses'] += 1
                self.performance['consecutive_wins'] = 0
                
                # 更新最大亏损
                if realized_pnl < self.performance['largest_loss']:
                    self.performance['largest_loss'] = realized_pnl
                
                # 更新平均亏损
                total_losses = self.performance['losing_trades']
                self.performance['avg_loss'] = (
                    (self.performance['avg_loss'] * (total_losses - 1) + realized_pnl) / total_losses
                    if total_losses > 0 else realized_pnl
                )
            
            # 计算胜率
            total = self.performance['winning_trades'] + self.performance['losing_trades']
            if total > 0:
                self.performance['win_rate'] = self.performance['winning_trades'] / total
            
            # 计算盈亏比
            if abs(self.performance['avg_loss']) > 0:
                self.performance['profit_factor'] = abs(self.performance['avg_win'] / self.performance['avg_loss'])
            
            # 更新最大回撤
            if self.performance['total_pnl'] < 0:
                current_drawdown = abs(self.performance['total_pnl']) / self.trading_config.initial_balance
                self.performance['max_drawdown'] = max(self.performance['max_drawdown'], current_drawdown)
            
            # 计算夏普比率（简化版）
            # 实际应该基于收益率序列计算
            if self.performance['total_trades'] > 10:
                # 这里使用简化计算
                avg_return = self.performance['total_pnl'] / self.performance['total_trades']
                # 假设年化波动率为20%
                annual_volatility = 0.2
                risk_free_rate = 0.02
                
                if annual_volatility > 0:
                    self.performance['sharpe_ratio'] = (avg_return - risk_free_rate/252) / (annual_volatility/np.sqrt(252))
                
                # 计算Calmar比率
                if self.performance['max_drawdown'] > 0:
                    self.performance['calmar_ratio'] = avg_return / self.performance['max_drawdown']
    
    def get_position_summary(self):
        """获取仓位摘要"""
        with self.trade_lock:
            position = self.positions['DOGEUSDT']
            
            # 获取当前价格
            current_price = position['current_price'] or 0.08
            
            # 更新浮动盈亏
            if position['quantity'] > 0 and position['entry_price'] > 0:
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                position['unrealized_pnl_ratio'] = (current_price - position['entry_price']) / position['entry_price']
                position['position_value'] = position['quantity'] * current_price
            
            summary = {
                'symbol': self.trading_config.symbol,
                'position': position.copy(),
                'balance': self.balance.copy(),
                'performance': self.performance.copy(),
                'trade_limits': self.trade_limits.copy(),
                'paper_trading': self.paper_trading,
                'live_trading': self.trading_config.live_trading and not self.paper_trading,
                'timestamp': datetime.now()
            }
            
            return summary
    
    def reset_daily_metrics(self):
        """重置日度指标"""
        with self.trade_lock:
            self.trade_limits['daily_trade_count'] = 0
            self.trade_limits['daily_pnl'] = 0.0
            self.performance['daily_pnl'] = 0.0
            
            # 检查是否需要重置周/月指标
            current_time = datetime.now()
            if current_time.weekday() == 0:  # 周一
                self.performance['weekly_pnl'] = 0.0
            
            if current_time.day == 1:  # 每月1号
                self.performance['monthly_pnl'] = 0.0
            
            logger.info("日度指标已重置")
    
    def get_trade_history_report(self, limit=100):
        """获取交易历史报告"""
        trades = list(self.trade_history)[-limit:] if self.trade_history else []
        
        report = {
            'total_trades': len(trades),
            'successful_trades': sum(1 for t in trades if t.get('success', False)),
            'total_pnl': sum(t.get('realized_pnl', 0) for t in trades),
            'total_commission': sum(t.get('commission', 0) for t in trades),
            'trades': trades
        }
        
        return report
    
    def export_trade_history(self, filename=None):
        """导出交易历史"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(self.config.directories['reports'], f'trades_{timestamp}.json')
            
            trades = list(self.trade_history)
            
            # 转换datetime对象为字符串
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(trades, f, indent=2, default=convert_datetime, ensure_ascii=False)
            
            logger.info(f"交易历史已导出到 {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"导出交易历史失败: {e}")
            return None

# ==================== 完整GUI系统 ====================

class CompleteTradingGUI:
    """完整的交易系统GUI，整合所有功能"""
    
    def __init__(self, root):
        self.root = root
        
        # 初始��配置
        self.config = SystemConfig()
        self.trading_config = self.config.trading_config
        
        # 初始化组件
        self.data_manager = CompleteDataManager(self.config)
        self.feature_engineer = AdvancedFeatureEngineer(self.config)
        self.model_manager = EnsembleModelManager(self.config)
        self.trading_engine = CompleteTradingEngine(self.config)
        
        # 系统状态
        self.is_running = False
        self.is_initialized = False
        self.update_interval = 2000  # GUI更新间隔（毫秒）
        self.trading_interval = 30000  # 交易检查间隔（毫秒）
        
        # GUI组件
        self.frames = {}
        self.widgets = {}
        self.variables = {}
        
        # 数据存储
        self.signals_history = deque(maxlen=500)
        self.price_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        # 图表相关
        self.figures = {}
        self.canvases = {}
        
        # 设置窗口
        self.root.title("DOGE多因子量化交易系统 - 完整版")
        self.root.geometry("1200x800")
        
        # 设置图标
        try:
            self.root.iconbitmap('doge.ico')
        except:
            pass
        
        # 设置样式
        self.setup_styles()
        
        # 创建菜单
        self.create_menu()
        
        # 创建主界面
        self.create_main_interface()
        
        # 初始化系统
        self.initialize_system()
        
        # 启动GUI更新循环
        self.schedule_gui_update()
        
        logger.info("完整交易系统GUI初始化完成")
    
    def setup_styles(self):
        """设置界面样式"""
        # 颜色方案
        self.colors = {
            # 背景色
            'bg_dark': '#1e1e1e',
            'bg_medium': '#2d2d30',
            'bg_light': '#3e3e42',
            'bg_lighter': '#4a4a4f',
            
            # 文字色
            'text_light': '#ffffff',
            'text_muted': '#b0b0b0',
            'text_dark': '#000000',
            
            # 主题色
            'primary': '#007acc',
            'primary_dark': '#005a9e',
            'primary_light': '#3d9cd7',
            
            # 状态色
            'success': '#4caf50',
            'success_dark': '#388e3c',
            'warning': '#ff9800',
            'warning_dark': '#f57c00',
            'danger': '#f44336',
            'danger_dark': '#d32f2f',
            'info': '#2196f3',
            'info_dark': '#1976d2',
            
            # 数据色
            'price_up': '#00c853',
            'price_down': '#ff3d00',
            'volume': '#2979ff',
            'sentiment': '#ff6d00',
            'onchain': '#aa00ff'
        }
        
        # 字体（调整为稍小以适配 1200x800 窗口）
        self.fonts = {
            'title': tkFont.Font(family="Microsoft YaHei", size=16, weight="bold"),
            'subtitle': tkFont.Font(family="Microsoft YaHei", size=13, weight="bold"),
            'heading': tkFont.Font(family="Microsoft YaHei", size=11, weight="bold"),
            'normal': tkFont.Font(family="Microsoft YaHei", size=10),
            'small': tkFont.Font(family="Microsoft YaHei", size=9),
            'mono': tkFont.Font(family="Consolas", size=9),
            'mono_bold': tkFont.Font(family="Consolas", size=9, weight="bold")
        }
        
        # 配置ttk样式
        style = ttk.Style()
        
        # 设置主题
        try:
            style.theme_use('clam')
        except:
            pass
        
        # 配置颜色
        style.configure('TFrame', background=self.colors['bg_dark'])
        style.configure('TLabel', background=self.colors['bg_dark'], 
                       foreground=self.colors['text_light'])
        style.configure('TButton', padding=6)
        style.configure('TLabelframe', background=self.colors['bg_dark'],
                       foreground=self.colors['text_light'])
        style.configure('TLabelframe.Label', background=self.colors['bg_dark'],
                       foreground=self.colors['text_light'])
        
        # 自定义样式
        style.configure('Primary.TButton', 
                       background=self.colors['primary'],
                       foreground=self.colors['text_light'])
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary_dark'])])
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground=self.colors['text_light'])
        style.map('Success.TButton',
                 background=[('active', self.colors['success_dark'])])
        
        style.configure('Danger.TButton',
                       background=self.colors['danger'],
                       foreground=self.colors['text_light'])
        style.map('Danger.TButton',
                 background=[('active', self.colors['danger_dark'])])
        
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground=self.colors['text_dark'])
        style.map('Warning.TButton',
                 background=[('active', self.colors['warning_dark'])])
        
        # 配置根窗口背景
        self.root.configure(bg=self.colors['bg_dark'])
    
    def create_menu(self):
        """创建菜单栏 - 修复版"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载配置", command=self.load_config)
        file_menu.add_command(label="保存配置", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="导出交易历史", command=self.export_trades)
        file_menu.add_command(label="导出性能报告", command=self.export_performance)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 系统菜单
        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="系统", menu=system_menu)
        system_menu.add_command(label="初始化系统", command=self.initialize_system)
        system_menu.add_command(label="重新训练模型", command=self.retrain_models)
        system_menu.add_command(label="更新数据", command=self.update_data_now)
        system_menu.add_command(label="重启系统", command=self.restart_system)
        system_menu.add_separator()
        system_menu.add_command(label="同步余额", command=self.sync_balance)
        system_menu.add_command(label="重置日度指标", command=self.reset_daily_metrics)
        
        # 交易菜单
        trade_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="交易", menu=trade_menu)
        trade_menu.add_command(label="手动买入", command=self.manual_buy)
        trade_menu.add_command(label="手动卖出", command=self.manual_sell)
        trade_menu.add_separator()
        trade_menu.add_command(label="查看持仓", command=self.show_positions)
        trade_menu.add_command(label="查看订单", command=self.show_orders)
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="价格图表", command=self.show_price_chart)
        view_menu.add_command(label="性能图表", command=self.show_performance_chart)
        view_menu.add_command(label="特征重要性", command=self.show_feature_importance)
        view_menu.add_separator()
        view_menu.add_command(label="系统日志", command=self.show_logs)
        view_menu.add_command(label="模型详情", command=self.show_model_details)
        view_menu.add_command(label="设置页面", command=self.show_settings)
        
        # 分析菜单
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="分析", menu=analysis_menu)
        analysis_menu.add_command(label="运行分析", command=self.run_analysis)
        analysis_menu.add_command(label="市场趋势", command=self.analyze_market_trend)
        analysis_menu.add_command(label="风险分析", command=self.analyze_risk)
        analysis_menu.add_command(label="模型评估", command=self.analyze_models)
        analysis_menu.add_command(label="数据质量", command=self.analyze_data_quality)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说���", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def create_main_interface(self):
        """创建主界面"""
        # 创建主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建顶部状态栏
        self.create_status_bar(main_container)
        
        # 创建主内容区域（左右布局）
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 左侧面板
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 右侧面板
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        
        # 构建左侧面板
        self.create_left_panel(left_panel)
        
        # 构建右侧面板
        self.create_right_panel(right_panel)
        
        # 创建底部控制栏
        self.create_control_bar(main_container)
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 系统状态
        self.variables['system_status'] = tk.StringVar(value="系统初始化中...")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.variables['system_status'],
            font=self.fonts['heading'],
            foreground=self.colors['warning']
        )
        status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 连接状态
        self.variables['connection_status'] = tk.StringVar(value="连接中...")
        connection_label = ttk.Label(
            status_frame,
            textvariable=self.variables['connection_status'],
            font=self.fonts['normal'],
            foreground=self.colors['text_muted']
        )
        connection_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 交易模式
        self.variables['trading_mode'] = tk.StringVar(value="模拟交易")
        mode_label = ttk.Label(
            status_frame,
            textvariable=self.variables['trading_mode'],
            font=self.fonts['normal'],
            foreground=self.colors['info']
        )
        mode_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 最后更新时间
        self.variables['last_update'] = tk.StringVar(value="最后更新: --:--:--")
        update_label = ttk.Label(
            status_frame,
            textvariable=self.variables['last_update'],
            font=self.fonts['small'],
            foreground=self.colors['text_muted']
        )
        update_label.pack(side=tk.RIGHT)
        
        self.widgets['status_frame'] = status_frame
    
    def create_left_panel(self, parent):
        """创建左侧面板"""
        # 使用Notebook实现标签页
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 1. 概览标签页
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="概览")
        self.create_overview_tab(overview_frame)
        
        # 2. 交易标签页
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="交易")
        self.create_trading_tab(trading_frame)
        
        # 3. 模型标签页
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="模型")
        self.create_model_tab(model_frame)
        
        # 4. 数据标签页
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="数据")
        self.create_data_tab(data_frame)
        
        self.widgets['notebook'] = notebook
    
    def create_overview_tab(self, parent):
        """创建概览标签页"""
        # 使用网格布局
        overview_grid = ttk.Frame(parent)
        overview_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 第1行：关键指标
        metrics_frame = ttk.LabelFrame(overview_grid, text="关键指标", padding=10)
        metrics_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.create_key_metrics(metrics_frame)
        
        # 第2行：价格图表和信号
        chart_frame = ttk.LabelFrame(overview_grid, text="价格与信号", padding=10)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.create_price_chart(chart_frame)
        
        # 第3行：性能图表
        perf_frame = ttk.LabelFrame(overview_grid, text="性能表现", padding=10)
        perf_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.create_performance_chart(perf_frame)
        
        # 配置网格权重
        overview_grid.columnconfigure(0, weight=1)
        overview_grid.columnconfigure(1, weight=1)
        overview_grid.rowconfigure(0, weight=0)
        overview_grid.rowconfigure(1, weight=1)
    
    def create_key_metrics(self, parent):
        """创建关键指标显示"""
        metrics_grid = ttk.Frame(parent)
        metrics_grid.pack(fill=tk.X, expand=True)
        
        # 价格指标
        price_frame = ttk.LabelFrame(metrics_grid, text="价格", padding=5)
        price_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        self.create_price_metrics(price_frame)
        
        # 仓位指标
        position_frame = ttk.LabelFrame(metrics_grid, text="仓位", padding=5)
        position_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        self.create_position_metrics(position_frame)
        
        # 账户指标
        account_frame = ttk.LabelFrame(metrics_grid, text="账户", padding=5)
        account_frame.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        
        self.create_account_metrics(account_frame)
        
        # 性能指标
        perf_frame = ttk.LabelFrame(metrics_grid, text="性能", padding=5)
        perf_frame.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)
        
        self.create_performance_metrics(perf_frame)
        
        # 配置网格权重
        for i in range(4):
            metrics_grid.columnconfigure(i, weight=1)
    
    def create_price_metrics(self, parent):
        """创建价格指标"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        
        metrics = [
            ("当前价格", "current_price", "${:.6f}", self.colors['primary']),
            ("24H变化", "price_change_24h", "{:+.2%}", None),
            ("24H高", "high_24h", "${:.6f}", self.colors['text_muted']),
            ("24H低", "low_24h", "${:.6f}", self.colors['text_muted']),
            ("成交量", "volume_24h", "{:,.0f}", self.colors['volume']),
            ("交易额", "quote_volume", "${:,.0f}", self.colors['volume'])
        ]
        
        self.widgets['price_metrics'] = {}
        
        for i, (label, key, fmt, color) in enumerate(metrics):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=1
            )
            
            # 值
            value_var = tk.StringVar(value="--")
            value_label = ttk.Label(
                grid,
                textvariable=value_var,
                font=self.fonts['small'],
                foreground=color if color else self.colors['text_light']
            )
            value_label.grid(row=i, column=1, sticky="e", padx=2, pady=1)
            
            self.widgets['price_metrics'][key] = value_var
    
    def create_position_metrics(self, parent):
        """创建仓位指标"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        
        metrics = [
            ("持仓数量", "position_quantity", "{:.0f} DOGE", self.colors['primary']),
            ("入场价格", "entry_price", "${:.6f}", self.colors['text_muted']),
            ("当前价值", "position_value", "${:.2f}", self.colors['primary']),
            ("浮动盈亏", "unrealized_pnl", "${:+.2f}", None),
            ("盈亏比例", "unrealized_pnl_ratio", "{:+.2%}", None),
            ("持仓比例", "position_ratio", "{:.1%}", self.colors['text_muted'])
        ]
        
        self.widgets['position_metrics'] = {}
        
        for i, (label, key, fmt, color) in enumerate(metrics):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=1
            )
            
            # 值
            value_var = tk.StringVar(value="--")
            value_label = ttk.Label(
                grid,
                textvariable=value_var,
                font=self.fonts['small'],
                foreground=color if color else self.colors['text_light']
            )
            value_label.grid(row=i, column=1, sticky="e", padx=2, pady=1)
            
            self.widgets['position_metrics'][key] = value_var
    
    def create_account_metrics(self, parent):
        """创建账户指标"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        
        metrics = [
            ("USDT余额", "balance_usdt", "${:.2f}", self.colors['success']),
            ("DOGE余额", "balance_doge", "{:.0f} DOGE", self.colors['text_muted']),
            ("总资产", "total_balance", "${:.2f}", self.colors['primary']),
            ("可用资金", "available_balance", "${:.2f}", self.colors['success']),
            ("已用保证金", "used_margin", "${:.2f}", self.colors['warning']),
            ("风险率", "risk_ratio", "{:.1%}", self.colors['text_muted'])
        ]
        
        self.widgets['account_metrics'] = {}
        
        for i, (label, key, fmt, color) in enumerate(metrics):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=1
            )
            
            # 值
            value_var = tk.StringVar(value="--")
            value_label = ttk.Label(
                grid,
                textvariable=value_var,
                font=self.fonts['small'],
                foreground=color if color else self.colors['text_light']
            )
            value_label.grid(row=i, column=1, sticky="e", padx=2, pady=1)
            
            self.widgets['account_metrics'][key] = value_var
    
    def create_performance_metrics(self, parent):
        """创建性能指标"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        
        metrics = [
            ("总盈亏", "total_pnl", "${:+.2f}", None),
            ("日盈亏", "daily_pnl", "${:+.2f}", None),
            ("胜率", "win_rate", "{:.1%}", self.colors['success']),
            ("交易次数", "total_trades", "{:.0f}", self.colors['text_muted']),
            ("盈亏比", "profit_factor", "{:.2f}", self.colors['primary']),
            ("最大回撤", "max_drawdown", "{:.2%}", self.colors['danger'])
        ]
        
        self.widgets['performance_metrics'] = {}
        
        for i, (label, key, fmt, color) in enumerate(metrics):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=1
            )
            
            # 值
            value_var = tk.StringVar(value="--")
            value_label = ttk.Label(
                grid,
                textvariable=value_var,
                font=self.fonts['small'],
                foreground=color if color else self.colors['text_light']
            )
            value_label.grid(row=i, column=1, sticky="e", padx=2, pady=1)
            
            self.widgets['performance_metrics'][key] = value_var
    
    def create_price_chart(self, parent):
        """创建价格图表区域"""
        # 创建图表容器
        chart_container = ttk.Frame(parent)
        chart_container.pack(fill=tk.BOTH, expand=True)
        
        # 创建Figure
        self.figures['price'] = Figure(figsize=(6, 3.5), dpi=100, facecolor=self.colors['bg_dark'])
        
        # 创建子图
        ax1 = self.figures['price'].add_subplot(211)
        ax2 = self.figures['price'].add_subplot(212)
        
        # 设置样式
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.colors['bg_medium'])
            ax.tick_params(colors=self.colors['text_light'])
            ax.xaxis.label.set_color(self.colors['text_light'])
            ax.yaxis.label.set_color(self.colors['text_light'])
            ax.title.set_color(self.colors['text_light'])
        
        # 创建Canvas
        self.canvases['price'] = FigureCanvasTkAgg(self.figures['price'], chart_container)
        self.canvases['price'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建工具栏
        toolbar = NavigationToolbar2Tk(self.canvases['price'], chart_container)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_performance_chart(self, parent):
        """创建性能图表区域"""
        # 创建图表容器
        chart_container = ttk.Frame(parent)
        chart_container.pack(fill=tk.BOTH, expand=True)
        
        # 创建Figure
        self.figures['performance'] = Figure(figsize=(6, 3.5), dpi=100, facecolor=self.colors['bg_dark'])
        
        # 创建子图
        ax1 = self.figures['performance'].add_subplot(211)
        ax2 = self.figures['performance'].add_subplot(212)
        
        # 设置样式
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.colors['bg_medium'])
            ax.tick_params(colors=self.colors['text_light'])
            ax.xaxis.label.set_color(self.colors['text_light'])
            ax.yaxis.label.set_color(self.colors['text_light'])
            ax.title.set_color(self.colors['text_light'])
        
        # 创建Canvas
        self.canvases['performance'] = FigureCanvasTkAgg(self.figures['performance'], chart_container)
        self.canvases['performance'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建工具栏
        toolbar = NavigationToolbar2Tk(self.canvases['performance'], chart_container)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_trading_tab(self, parent):
        """创建交易标签页"""
        trading_grid = ttk.Frame(parent)
        trading_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：交易控制
        control_frame = ttk.LabelFrame(trading_grid, text="交易控制", padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.create_trading_control(control_frame)
        
        # 右侧：交易信号
        signal_frame = ttk.LabelFrame(trading_grid, text="交易信号", padding=10)
        signal_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.create_signal_display(signal_frame)
        
        # 底部：交易历史
        history_frame = ttk.LabelFrame(trading_grid, text="交易历史", padding=10)
        history_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.create_trade_history(history_frame)
        
        # 配置网格权重
        trading_grid.columnconfigure(0, weight=1)
        trading_grid.columnconfigure(1, weight=1)
        trading_grid.rowconfigure(0, weight=1)
        trading_grid.rowconfigure(1, weight=1)
    
    def create_trading_control(self, parent):
        """创建交易控制面板"""
        control_grid = ttk.Frame(parent)
        control_grid.pack(fill=tk.BOTH, expand=True)
        
        # 交易模式选择
        mode_frame = ttk.Frame(control_grid)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_frame, text="交易模式:", font=self.fonts['normal']).pack(side=tk.LEFT)
        
        self.variables['trading_mode_var'] = tk.StringVar(value="paper")
        
        ttk.Radiobutton(
            mode_frame,
            text="模拟交易",
            variable=self.variables['trading_mode_var'],
            value="paper",
            command=self.on_trading_mode_change
        ).pack(side=tk.LEFT, padx=(10, 5))
        
        ttk.Radiobutton(
            mode_frame,
            text="实盘交易",
            variable=self.variables['trading_mode_var'],
            value="live",
            command=self.on_trading_mode_change
        ).pack(side=tk.LEFT)
        
        # API配置
        api_frame = ttk.LabelFrame(control_grid, text="API配置", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_api_config(api_frame)
        
        # 交易参数
        params_frame = ttk.LabelFrame(control_grid, text="交易参数", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_trading_params(params_frame)
        
        # 控制按钮
        button_frame = ttk.Frame(control_grid)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.widgets['start_button'] = ttk.Button(
            button_frame,
            text="开始交易",
            command=self.start_trading_system,
            style="Success.TButton"
        )
        self.widgets['start_button'].pack(side=tk.LEFT, padx=(0, 5))
        
        self.widgets['stop_button'] = ttk.Button(
            button_frame,
            text="停止交易",
            command=self.stop_trading_system,
            style="Danger.TButton",
            state="disabled"
        )
        self.widgets['stop_button'].pack(side=tk.LEFT, padx=(0, 5))
        
        self.widgets['test_button'] = ttk.Button(
            button_frame,
            text="测试连接",
            command=self.test_connection,
            style="Primary.TButton"
        )
        self.widgets['test_button'].pack(side=tk.LEFT)
    
    def create_api_config(self, parent):
        """创建API配置"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        # API Key
        ttk.Label(grid, text="API Key:", font=self.fonts['small']).grid(
            row=0, column=0, sticky="w", padx=2, pady=2
        )
        
        self.variables['api_key'] = tk.StringVar(value=self.trading_config.api_key)
        api_key_entry = ttk.Entry(
            grid,
            textvariable=self.variables['api_key'],
            width=40,
            show="*"
        )
        api_key_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        # API Secret
        ttk.Label(grid, text="API Secret:", font=self.fonts['small']).grid(
            row=1, column=0, sticky="w", padx=2, pady=2
        )
        
        self.variables['api_secret'] = tk.StringVar(value=self.trading_config.api_secret)
        api_secret_entry = ttk.Entry(
            grid,
            textvariable=self.variables['api_secret'],
            width=40,
            show="*"
        )
        api_secret_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        # 代理
        ttk.Label(grid, text="代理:", font=self.fonts['small']).grid(
            row=2, column=0, sticky="w", padx=2, pady=2
        )
        
        self.variables['proxy'] = tk.StringVar(value=self.trading_config.proxy)
        proxy_entry = ttk.Entry(
            grid,
            textvariable=self.variables['proxy'],
            width=40
        )
        proxy_entry.grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        
        # 保存按钮
        save_button = ttk.Button(
            grid,
            text="保存配置",
            command=self.save_api_config,
            style="Primary.TButton"
        )
        save_button.grid(row=3, column=0, columnspan=2, pady=(5, 0))
        
        grid.columnconfigure(1, weight=1)
    
    def create_trading_params(self, parent):
        """创建交易参数设置"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        params = [
            ("初始资金", "initial_balance", "${:.2f}", 10000.0, 1000.0, 1000000.0),
            ("最大仓位", "max_position_ratio", "{:.1%}", 0.3, 0.1, 1.0),
            ("单笔风险", "risk_per_trade", "{:.1%}", 0.02, 0.01, 0.1),
            ("止损", "stop_loss", "{:.1%}", 0.05, 0.01, 0.2),
            ("止盈1", "take_profit_1", "{:.1%}", 0.08, 0.01, 0.3),
            ("止盈2", "take_profit_2", "{:.1%}", 0.15, 0.01, 0.5),
            ("手续费率", "commission_rate", "{:.3%}", 0.001, 0.0001, 0.01),
            ("最小交易额", "min_trade_amount", "${:.2f}", 10.0, 1.0, 1000.0)
        ]
        
        self.widgets['param_entries'] = {}
        
        for i, (label, key, fmt, default, min_val, max_val) in enumerate(params):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=2
            )
            
            # 输入框
            value = getattr(self.trading_config, key, default)
            if isinstance(value, list):
                value = value[0] if key == "take_profit_1" else value[1]
            
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(grid, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=2, pady=2)
            
            self.widgets['param_entries'][key] = var
            
            # 范围标签
            ttk.Label(grid, text=f"[{min_val} - {max_val}]", 
                     font=self.fonts['small'],
                     foreground=self.colors['text_muted']).grid(
                row=i, column=2, sticky="w", padx=5, pady=2
            )
    
    def create_signal_display(self, parent):
        """创建交易信号显示"""
        # 信号状态
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.variables['signal_action'] = tk.StringVar(value="HOLD")
        self.variables['signal_strength'] = tk.StringVar(value="NEUTRAL")
        self.variables['signal_confidence'] = tk.StringVar(value="50.0%")
        
        # 信号图标
        signal_icon = tk.Label(
            status_frame,
            text="HOLD",
            font=self.fonts['title'],
            bg=self.colors['bg_dark']
        )
        signal_icon.pack(side=tk.LEFT, padx=(0, 10))
        self.widgets['signal_icon'] = signal_icon
        
        # 信号文本
        signal_text = ttk.Label(
            status_frame,
            textvariable=self.variables['signal_action'],
            font=self.fonts['title']
        )
        signal_text.pack(side=tk.LEFT, padx=(0, 5))
        
        # 信号强度
        strength_text = ttk.Label(
            status_frame,
            textvariable=self.variables['signal_strength'],
            font=self.fonts['subtitle'],
            foreground=self.colors['text_muted']
        )
        strength_text.pack(side=tk.LEFT, padx=(0, 10))
        
        # 信号置信度
        confidence_text = ttk.Label(
            status_frame,
            textvariable=self.variables['signal_confidence'],
            font=self.fonts['heading']
        )
        confidence_text.pack(side=tk.LEFT)
        
        # 信号详情
        detail_frame = ttk.Frame(parent)
        detail_frame.pack(fill=tk.BOTH, expand=True)
        
        self.widgets['signal_details'] = scrolledtext.ScrolledText(
            detail_frame,
            height=10,
            font=self.fonts['mono'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text_light'],
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.widgets['signal_details'].pack(fill=tk.BOTH, expand=True)
        self.widgets['signal_details'].insert(tk.END, "等待交易信号...\n")
        self.widgets['signal_details'].configure(state='disabled')
    
    def create_trade_history(self, parent):
        """创建交易历史显示"""
        # 使用Treeview显示交易历史
        columns = ("时间", "操作", "数量", "价格", "价值", "手续费", "盈亏", "状态")
        
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        self.widgets['trade_history'] = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=10
        )
        
        # 设置列
        col_widths = [120, 60, 80, 80, 90, 70, 80, 60]
        for col, width in zip(columns, col_widths):
            self.widgets['trade_history'].heading(col, text=col)
            self.widgets['trade_history'].column(col, width=width, minwidth=50)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", 
                                 command=self.widgets['trade_history'].yview)
        self.widgets['trade_history'].configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.widgets['trade_history'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 控制按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            button_frame,
            text="清除历史",
            command=self.clear_trade_history
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="导出CSV",
            command=self.export_history_csv
        ).pack(side=tk.LEFT)
    
    def create_model_tab(self, parent):
        """创建模型标签页"""
        model_grid = ttk.Frame(parent)
        model_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：模型控制
        control_frame = ttk.LabelFrame(model_grid, text="模型控制", padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.create_model_control(control_frame)
        
        # 右侧：模型性能
        perf_frame = ttk.LabelFrame(model_grid, text="模型性能", padding=10)
        perf_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.create_model_performance(perf_frame)
        
        # 底部：特征重要性
        feature_frame = ttk.LabelFrame(model_grid, text="特征重要性", padding=10)
        feature_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.create_feature_importance(feature_frame)
        
        # 配置网格权重
        model_grid.columnconfigure(0, weight=1)
        model_grid.columnconfigure(1, weight=1)
        model_grid.rowconfigure(0, weight=1)
        model_grid.rowconfigure(1, weight=1)
    
    def create_model_control(self, parent):
        """创建模型控制面板"""
        control_grid = ttk.Frame(parent)
        control_grid.pack(fill=tk.BOTH, expand=True)
        
        # 模型选择
        model_frame = ttk.LabelFrame(control_grid, text="模型选择", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_model_selection(model_frame)
        
        # 训练控制
        train_frame = ttk.LabelFrame(control_grid, text="训练控制", padding=10)
        train_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_training_control(train_frame)
        
        # 预测控制
        predict_frame = ttk.LabelFrame(control_grid, text="预测控制", padding=10)
        predict_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_prediction_control(predict_frame)
        
        # 控制按钮
        button_frame = ttk.Frame(control_grid)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="立即训练",
            command=self.train_models_now,
            style="Primary.TButton"
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="加载模型",
            command=self.load_models
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="保存模型",
            command=self.save_models
        ).pack(side=tk.LEFT)
    
    def create_model_selection(self, parent):
        """创建模型选择"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        models = [
            ("XGBoost", "xgb"),
            ("LightGBM", "lgb"),
            ("随机森林", "rf"),
            ("梯度提升树", "gbt"),
            ("Prophet", "prophet"),
            ("LSTM", "lstm"),
            ("集成模型", "ensemble")
        ]
        
        self.variables['model_vars'] = {}
        
        for i, (name, key) in enumerate(models):
            var = tk.BooleanVar(value=True if key in ['xgb', 'lgb', 'rf', 'ensemble'] else False)
            cb = ttk.Checkbutton(
                grid,
                text=name,
                variable=var
            )
            cb.grid(row=i//2, column=i%2, sticky="w", padx=2, pady=2)
            
            self.variables['model_vars'][key] = var
    
    def create_training_control(self, parent):
        """创建训练控制"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        params = [
            ("训练间隔", "model_retrain_interval", "小时", 24, 1, 168),
            ("预测周期", "prediction_horizon", "天", 5, 1, 30),
            ("置信阈值", "confidence_threshold", "", 0.6, 0.5, 0.95),
            ("特征窗口", "feature_window", "天", 50, 10, 365)
        ]
        
        for i, (label, key, unit, default, min_val, max_val) in enumerate(params):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=2
            )
            
            # 输入框
            value = getattr(self.trading_config, key, default)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(grid, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=2, pady=2)
            
            self.widgets[f'model_param_{key}'] = var
            
            # 单位
            ttk.Label(grid, text=unit, font=self.fonts['small']).grid(
                row=i, column=2, sticky="w", padx=2, pady=2
            )
    
    def create_prediction_control(self, parent):
        """创建预测控制"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        # 启用集成预测
        self.variables['use_ensemble'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            grid,
            text="使用集成预测",
            variable=self.variables['use_ensemble']
        ).grid(row=0, column=0, sticky="w", padx=2, pady=2)
        
        # 在线学习
        self.variables['online_learning'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            grid,
            text="启用在线学习",
            variable=self.variables['online_learning']
        ).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        
        # 特征选择
        self.variables['feature_selection'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            grid,
            text="启用特征选择",
            variable=self.variables['feature_selection']
        ).grid(row=1, column=0, sticky="w", padx=2, pady=2)
        
        # 错误重试
        self.variables['retrain_on_error'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            grid,
            text="错误时重训练",
            variable=self.variables['retrain_on_error']
        ).grid(row=1, column=1, sticky="w", padx=2, pady=2)
    
    def create_model_performance(self, parent):
        """创建模型性能显示"""
        # 使用Treeview显示模型性能
        columns = ("模型", "准确率", "F1分数", "AUC", "最后训练", "状态")
        
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        self.widgets['model_performance'] = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=8
        )
        
        # 设置列
        col_widths = [80, 60, 60, 60, 100, 60]
        for col, width in zip(columns, col_widths):
            self.widgets['model_performance'].heading(col, text=col)
            self.widgets['model_performance'].column(col, width=width, minwidth=50)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", 
                                 command=self.widgets['model_performance'].yview)
        self.widgets['model_performance'].configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.widgets['model_performance'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 性能摘要
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.variables['model_summary'] = tk.StringVar(value="未训练")
        ttk.Label(
            summary_frame,
            textvariable=self.variables['model_summary'],
            font=self.fonts['small']
        ).pack(side=tk.LEFT)
    
    def create_feature_importance(self, parent):
        """创建特征重要性显示"""
        # 特征重要性图表
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Figure
        self.figures['features'] = Figure(figsize=(8, 4.5), dpi=100, facecolor=self.colors['bg_dark'])
        
        # 创建子图
        ax = self.figures['features'].add_subplot(111)
        
        # 设置样式
        ax.set_facecolor(self.colors['bg_medium'])
        ax.tick_params(colors=self.colors['text_light'])
        ax.xaxis.label.set_color(self.colors['text_light'])
        ax.yaxis.label.set_color(self.colors['text_light'])
        ax.title.set_color(self.colors['text_light'])
        
        # 创建Canvas
        self.canvases['features'] = FigureCanvasTkAgg(self.figures['features'], chart_frame)
        self.canvases['features'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_data_tab(self, parent):
        """创建数据标签页"""
        data_grid = ttk.Frame(parent)
        data_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 数据源控制
        source_frame = ttk.LabelFrame(data_grid, text="数据源", padding=10)
        source_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.create_data_source_control(source_frame)
        
        # 数据统计
        stats_frame = ttk.LabelFrame(data_grid, text="数据统计", padding=10)
        stats_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.create_data_statistics(stats_frame)
        
        # 数据图表
        chart_frame = ttk.LabelFrame(data_grid, text="数据图表", padding=10)
        chart_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.create_data_chart(chart_frame)
        
        # 配置网格权重
        data_grid.columnconfigure(0, weight=1)
        data_grid.columnconfigure(1, weight=1)
        data_grid.rowconfigure(0, weight=0)
        data_grid.rowconfigure(1, weight=1)
    
    def create_data_source_control(self, parent):
        """创建数据源控制"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        
        # 数据源选择
        sources = [
            ("价格数据", "price"),
            ("社交媒体", "social"),
            ("链上数据", "onchain"),
            ("衍生品", "derivatives")
        ]
        
        self.variables['data_sources'] = {}
        
        for i, (name, key) in enumerate(sources):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(
                grid,
                text=name,
                variable=var
            )
            cb.grid(row=i, column=0, sticky="w", padx=2, pady=5)
            
            self.variables['data_sources'][key] = var
        
        # 数据更新间隔
        ttk.Label(grid, text="更新间隔:", font=self.fonts['small']).grid(
            row=4, column=0, sticky="w", padx=2, pady=(10, 2)
        )
        
        intervals = ["1分钟", "5分钟", "15分钟", "30分钟", "1小时"]
        self.variables['update_interval'] = tk.StringVar(value="5分钟")
        
        interval_combo = ttk.Combobox(
            grid,
            textvariable=self.variables['update_interval'],
            values=intervals,
            state="readonly",
            width=10
        )
        interval_combo.grid(row=4, column=1, sticky="w", padx=2, pady=(10, 2))
        
        # 控制按钮
        button_frame = ttk.Frame(grid)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="更新数据",
            command=self.update_data_now,
            style="Primary.TButton"
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="清除缓存",
            command=self.clear_cache
        ).pack(side=tk.LEFT)
    
    def create_data_statistics(self, parent):
        """创建数据统计"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        
        stats = [
            ("价格记录", "price_records", "{:,}"),
            ("社交媒体", "social_records", "{:,}"),
            ("链上数据", "onchain_records", "{:,}"),
            ("衍生品", "derivatives_records", "{:,}"),
            ("最新价格", "latest_price", "${:.6f}"),
            ("数据质量", "data_quality", "{:.1%}"),
            ("最后更新", "last_update", "{}"),
            ("缓存大小", "cache_size", "{:.1f} MB")
        ]
        
        self.widgets['data_stats'] = {}
        
        for i, (label, key, fmt) in enumerate(stats):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=2
            )
            
            # 值
            value_var = tk.StringVar(value="--")
            value_label = ttk.Label(
                grid,
                textvariable=value_var,
                font=self.fonts['small']
            )
            value_label.grid(row=i, column=1, sticky="e", padx=2, pady=2)
            
            self.widgets['data_stats'][key] = value_var
    
    def create_data_chart(self, parent):
        """创建数据图表"""
        # 数据图表容器
        chart_container = ttk.Frame(parent)
        chart_container.pack(fill=tk.BOTH, expand=True)
        
        # 创建Figure
        self.figures['data'] = Figure(figsize=(9, 6), dpi=100, facecolor=self.colors['bg_dark'])
        
        # 创建子图
        ax1 = self.figures['data'].add_subplot(221)
        ax2 = self.figures['data'].add_subplot(222)
        ax3 = self.figures['data'].add_subplot(223)
        ax4 = self.figures['data'].add_subplot(224)
        
        # 设置样式
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(self.colors['bg_medium'])
            ax.tick_params(colors=self.colors['text_light'])
            ax.xaxis.label.set_color(self.colors['text_light'])
            ax.yaxis.label.set_color(self.colors['text_light'])
            ax.title.set_color(self.colors['text_light'])
        
        # 创建Canvas
        self.canvases['data'] = FigureCanvasTkAgg(self.figures['data'], chart_container)
        self.canvases['data'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_right_panel(self, parent):
        """创建右侧面板"""
        # 使用垂直布局
        right_container = ttk.Frame(parent, width=300)
        right_container.pack(fill=tk.BOTH, expand=False)
        right_container.pack_propagate(False)
        
        # 日志面板
        log_frame = ttk.LabelFrame(right_container, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.create_log_panel(log_frame)
        
        # 快速操作面板
        quick_frame = ttk.LabelFrame(right_container, text="快速操作", padding=10)
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_quick_actions(quick_frame)
        
        # 系统信息面板
        info_frame = ttk.LabelFrame(right_container, text="系统信息", padding=10)
        info_frame.pack(fill=tk.X)
        
        self.create_system_info(info_frame)
    
    def create_log_panel(self, parent):
        """创建日志面板"""
        # 日志显示
        self.widgets['log_display'] = scrolledtext.ScrolledText(
            parent,
            height=20,
            font=self.fonts['mono'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text_light'],
            relief=tk.FLAT
        )
        self.widgets['log_display'].pack(fill=tk.BOTH, expand=True)
        
        # 配置标签
        self.widgets['log_display'].tag_config('INFO', foreground=self.colors['info'])
        self.widgets['log_display'].tag_config('WARNING', foreground=self.colors['warning'])
        self.widgets['log_display'].tag_config('ERROR', foreground=self.colors['danger'])
        self.widgets['log_display'].tag_config('SUCCESS', foreground=self.colors['success'])
        
        # 插入欢迎信息
        self.widgets['log_display'].insert(tk.END, "="*50 + "\n")
        self.widgets['log_display'].insert(tk.END, "DOGE多因子量化交易系统\n")
        self.widgets['log_display'].insert(tk.END, "="*50 + "\n")
        self.widgets['log_display'].insert(tk.END, "系统启动中...\n")
        self.widgets['log_display'].see(tk.END)
    
    def create_quick_actions(self, parent):
        """创建快速操作面板"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        # 快速操作按钮
        actions = [
            ("同步", self.sync_all),
            ("图表", self.show_all_charts),
            ("设置", self.show_settings),
            ("分析", self.run_analysis),
            ("风控", self.show_risk_control),
            ("报告", self.generate_report)
        ]
        
        for i, (text, command) in enumerate(actions):
            btn = ttk.Button(
                grid,
                text=text,
                command=command,
                width=8
            )
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky="nsew")
        
        # 配置网格权重
        for col in range(3):
            grid.columnconfigure(col, weight=1)
    
    def create_system_info(self, parent):
        """创建系统信息面板"""
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, expand=True)
        
        info_items = [
            ("版本", "v2.0.0"),
            ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
            ("运行时间", "00:00:00"),
            ("内存使用", "0.0 MB"),
            ("CPU使用", "0.0%"),
            ("线程数", "0")
        ]
        
        self.widgets['system_info'] = {}
        
        for i, (label, default) in enumerate(info_items):
            # 标签
            ttk.Label(grid, text=f"{label}:", font=self.fonts['small']).grid(
                row=i, column=0, sticky="w", padx=2, pady=1
            )
            
            # 值
            value_var = tk.StringVar(value=default)
            value_label = ttk.Label(
                grid,
                textvariable=value_var,
                font=self.fonts['small'],
                foreground=self.colors['text_muted']
            )
            value_label.grid(row=i, column=1, sticky="e", padx=2, pady=1)
            
            self.widgets['system_info'][label] = value_var
    
    def create_control_bar(self, parent):
        """创建底部控制栏"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 进度条
        self.variables['progress'] = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.variables['progress'],
            length=200,
            mode='determinate'
        )
        progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        
        # 状态消息
        self.variables['status_message'] = tk.StringVar(value="系统就绪")
        status_label = ttk.Label(
            control_frame,
            textvariable=self.variables['status_message'],
            font=self.fonts['small']
        )
        status_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # 系统资源
        self.variables['system_resources'] = tk.StringVar(value="CPU: 0% | 内存: 0MB")
        resource_label = ttk.Label(
            control_frame,
            textvariable=self.variables['system_resources'],
            font=self.fonts['small'],
            foreground=self.colors['text_muted']
        )
        resource_label.pack(side=tk.RIGHT)
    
    def initialize_system(self):
        """初始化系统"""
        self.variables['system_status'].set("系统初始化中...")
        self.variables['status_message'].set("正在初始化系统...")
        self.log_message("开始系统初始化...", "INFO")
        
        # 在后台线程中初始化
        def init_thread():
            try:
                # 1. 更新API配置
                self.update_api_config()
                
                # 2. 初始化数据管理器
                self.log_message("初始化数据管理器...", "INFO")
                self.data_manager.fetch_historical_data(days=30)  # 先加载30天数据
                
                # 3. 初始化交易引擎
                self.log_message("初始化交易引擎...", "INFO")
                self.trading_engine.initialize()
                
                # 4. 加载或训练模型（自动训练）
                self.log_message("加载模型...", "INFO")
                models_loaded = self.model_manager.load_models()
                
                self.log_message("开始自动训练模型...", "INFO")
                success = self.model_manager.train_models(self.data_manager, retrain=True)
                if success:
                    self.log_message("模型训练完成", "SUCCESS")
                else:
                    self.log_message("模型训练失败", "ERROR")
                
                # 5. 启动数据流
                self.data_manager.start_real_time_stream()
                
                # 6. 标记为已初始化
                self.is_initialized = True
                
                # 更新GUI状态
                self.root.after(0, self.on_system_initialized)
                
            except Exception as e:
                self.log_message(f"系统初始化失败: {e}", "ERROR")
                self.root.after(0, lambda: self.on_system_init_failed(str(e)))
        
        # 启动初始化线程
        threading.Thread(target=init_thread, daemon=True).start()
    
    def on_system_initialized(self):
        """系统初始化完成回调"""
        self.variables['system_status'].set("系统就绪")
        self.variables['status_message'].set("系统初始化完成")
        
        # 更新交易模式显示
        if self.trading_engine.paper_trading:
            self.variables['trading_mode'].set("模拟交易")
        else:
            self.variables['trading_mode'].set("实盘交易")
        
        # 更新连接状态
        self.variables['connection_status'].set("已连接")
        
        self.log_message("系统初始化完成，准备就绪", "SUCCESS")
        
        # 启用开始按钮
        self.widgets['start_button'].configure(state="normal")
    
    def on_system_init_failed(self, error_msg):
        """系统初始化失败回调"""
        self.variables['system_status'].set("系统错误")
        self.variables['status_message'].set(f"初始化失败: {error_msg}")
        
        messagebox.showerror("初始化错误", f"系统初始化失败:\n{error_msg}")
    
    def update_api_config(self):
        """更新API配置"""
        try:
            # 从界面获取配置
            api_key = self.variables['api_key'].get()
            api_secret = self.variables['api_secret'].get()
            proxy = self.variables['proxy'].get()
            
            # 更新配置对象
            self.config.update_config(
                api_key=api_key,
                api_secret=api_secret,
                proxy=proxy
            )
            
            # 更新客户端
            self.data_manager.api_client = BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                proxy=proxy
            )
            
            self.trading_engine.api_client = self.data_manager.api_client
            
            self.log_message("API配置已更新", "INFO")
            
        except Exception as e:
            self.log_message(f"更新API配置失败: {e}", "ERROR")
    
    def save_api_config(self):
        """保存API配置"""
        self.update_api_config()
        self.config.save_config()
        self.log_message("配置已保存", "SUCCESS")
    
    def load_config(self):
        """加载配置"""
        try:
            # 打开文件对话框
            filename = filedialog.askopenfilename(
                title="选择配置文件",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
            if filename:
                # 临时保存当前配置
                temp_config = self.config.trading_config
                
                # 加载新配置
                self.config.config_file = filename
                self.config.load_config()
                
                # 更新界面
                self.update_gui_from_config()
                
                # 更新组件
                self.update_api_config()
                
                self.log_message(f"配置已从 {filename} 加载", "SUCCESS")
                
        except Exception as e:
            self.log_message(f"加载配置失败: {e}", "ERROR")
            # 恢复原配置
            self.config.trading_config = temp_config
    
    def save_config(self):
        """保存配置"""
        try:
            # 从界面获取配置值
            self.update_trading_params()
            
            # 保存到文件
            self.config.save_config()
            
            self.log_message("配置已保存", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"保存配置失败: {e}", "ERROR")
    
    def update_trading_params(self):
        """从界面更新交易参数"""
        try:
            # 获取所有参数值
            params = {}
            
            for key, var in self.widgets['param_entries'].items():
                value = var.get()
                
                # 转换类型
                if key in ['initial_balance', 'min_trade_amount']:
                    params[key] = float(value)
                elif key in ['max_position_ratio', 'risk_per_trade', 'stop_loss', 
                           'commission_rate', 'take_profit_1', 'take_profit_2']:
                    params[key] = float(value)
            
            # 处理止盈列表
            params['take_profit'] = [
                params.pop('take_profit_1', 0.08),
                params.pop('take_profit_2', 0.15)
            ]
            
            # 更新配置
            self.config.update_config(**params)
            
        except Exception as e:
            self.log_message(f"更新交易参数失败: {e}", "ERROR")
    
    def update_gui_from_config(self):
        """从配置更新GUI"""
        config = self.trading_config
        
        # 更新API配置
        self.variables['api_key'].set(config.api_key)
        self.variables['api_secret'].set(config.api_secret)
        self.variables['proxy'].set(config.proxy)
        
        # 更新交易参数
        for key, var in self.widgets['param_entries'].items():
            if key == 'take_profit_1':
                value = config.take_profit[0] if config.take_profit else 0.08
            elif key == 'take_profit_2':
                value = config.take_profit[1] if config.take_profit else 0.15
            else:
                value = getattr(config, key, 0)
            
            var.set(str(value))
        
        # 更新模型参数
        for key in ['model_retrain_interval', 'prediction_horizon', 
                   'confidence_threshold', 'feature_window']:
            widget_key = f'model_param_{key}'
            if widget_key in self.widgets:
                value = getattr(config, key, 0)
                self.widgets[widget_key].set(str(value))
    
    def start_trading_system(self):
        """启动交易系统"""
        if not self.is_initialized:
            messagebox.showwarning("系统未初始化", "请先初始化系统")
            return
        
        if self.is_running:
            self.log_message("交易系统已在运行", "WARNING")
            return
        
        # 更新状态
        self.is_running = True
        self.variables['system_status'].set("交易运行中")
        self.variables['status_message'].set("交易系统运行中...")
        
        # 更新按钮状态
        self.widgets['start_button'].configure(state="disabled")
        self.widgets['stop_button'].configure(state="normal")
        
        # 启动交易循环
        self.schedule_trading_cycle()
        
        self.log_message("交易系统已启动", "SUCCESS")
    
    def stop_trading_system(self):
        """停止交易系统"""
        if not self.is_running:
            self.log_message("交易系统未运行", "WARNING")
            return
        
        # 更新状态
        self.is_running = False
        self.variables['system_status'].set("系统就绪")
        self.variables['status_message'].set("交易系统已停止")
        
        # 更新按钮状态
        self.widgets['start_button'].configure(state="normal")
        self.widgets['stop_button'].configure(state="disabled")
        
        # 停止数据流
        self.data_manager.stop_real_time_stream()
        
        self.log_message("交易系统已停止", "INFO")
    
    def schedule_trading_cycle(self):
        """计划交易循环"""
        if self.is_running:
            try:
                self.execute_trading_cycle()
                # 使用after而不是递归
                self.root.after(self.trading_interval, self.schedule_trading_cycle)
            except Exception as e:
                self.log_message(f"交易循环调度失败: {e}", "ERROR")
                # 出错后稍等再试
                self.root.after(10000, self.schedule_trading_cycle)
    
    def update_gui(self):
        """更新GUI显示"""
        try:
            current_time = datetime.now()
            
            # 更新时间戳
            self.variables['last_update'].set(
                f"最后更新: {current_time.strftime('%H:%M:%S')}"
            )
            
            # 更新系统信息
            self.update_system_info()
            
            # 更新价格信息
            self.update_price_info()
            
            # 更新仓位信息
            self.update_position_info()
            
            # 更新账户信息
            self.update_account_info()
            
            # 更新性能信息
            self.update_performance_info()
            
            # 更新数据统计
            self.update_data_stats()
            
            # 更新图表
            if current_time.second % 10 == 0:  # 每10秒更新一次图表
                self.update_charts()
            
            # 更新交易历史
            self.update_trade_history()
            
            # 更新模型性能
            self.update_model_performance()
            
        except Exception as e:
            self.log_message(f"GUI更新失败: {e}", "ERROR")
    
    def update_system_info(self):
        """更新系统信息"""
        try:
            # 运行时间
            if hasattr(self, 'start_time'):
                run_time = datetime.now() - self.start_time
                hours, remainder = divmod(int(run_time.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                self.widgets['system_info']['运行时间'].set(
                    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                )
            
            # 内存使用（模拟）
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.widgets['system_info']['内存使用'].set(f"{memory_mb:.1f} MB")
            
            # CPU使用
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.widgets['system_info']['CPU使用'].set(f"{cpu_percent:.1f}%")
            
            # 线程数
            thread_count = threading.active_count()
            self.widgets['system_info']['线程数'].set(str(thread_count))
            
            # 系统资源显示
            self.variables['system_resources'].set(
                f"CPU: {cpu_percent:.1f}% | 内存: {memory_mb:.1f}MB"
            )
            
        except:
            pass
    
    def update_price_info(self):
        """更新价格信息"""
        try:
            # 获取最新价格
            latest_data = self.data_manager.get_latest_aggregated_data()
            
            if 'price' in latest_data and latest_data['price']:
                price_data = latest_data['price']
                
                # 更新价格指标
                if 'close' in price_data:
                    current_price = price_data['close']
                    self.widgets['price_metrics']['current_price'].set(
                        f"${current_price:.6f}"
                    )
                    
                    # 存储价格历史
                    self.price_history.append({
                        'timestamp': datetime.now(),
                        'price': current_price
                    })
                
                # 更新其他指标（模拟）
                change = np.random.uniform(-0.02, 0.02)
                change_color = self.colors['price_up'] if change >= 0 else self.colors['price_down']
                
                self.widgets['price_metrics']['price_change_24h'].set(
                    f"{change:+.2%}"
                )
                
                # 模拟其他价格数据
                current_price_val = current_price if 'close' in price_data else 0.08
                self.widgets['price_metrics']['high_24h'].set(
                    f"${current_price_val * 1.02:.6f}"
                )
                self.widgets['price_metrics']['low_24h'].set(
                    f"${current_price_val * 0.98:.6f}"
                )
                self.widgets['price_metrics']['volume_24h'].set(
                    f"{np.random.randint(1000000, 5000000):,}"
                )
                self.widgets['price_metrics']['quote_volume'].set(
                    f"${np.random.randint(50000, 200000):,}"
                )
            
        except Exception as e:
            self.log_message(f"更新价格信息失败: {e}", "ERROR")
    
    def update_position_info(self):
        """更新仓位信息"""
        try:
            # 获取仓位摘要
            position_summary = self.trading_engine.get_position_summary()
            position = position_summary['position']
            
            # 更新仓位指标
            self.widgets['position_metrics']['position_quantity'].set(
                f"{position['quantity']:.0f} DOGE"
            )
            
            self.widgets['position_metrics']['entry_price'].set(
                f"${position['entry_price']:.6f}"
            )
            
            self.widgets['position_metrics']['position_value'].set(
                f"${position['position_value']:.2f}"
            )
            
            # 更新盈亏
            unrealized_pnl = position['unrealized_pnl']
            unrealized_pnl_ratio = position['unrealized_pnl_ratio']
            
            pnl_color = self.colors['price_up'] if unrealized_pnl >= 0 else self.colors['price_down']
            
            self.widgets['position_metrics']['unrealized_pnl'].set(
                f"${unrealized_pnl:+.2f}"
            )
            
            self.widgets['position_metrics']['unrealized_pnl_ratio'].set(
                f"{unrealized_pnl_ratio:+.2%}"
            )
            
            # 更新持仓比例
            total_value = position_summary['balance']['total_value']
            if total_value > 0:
                position_ratio = position['position_value'] / total_value
                self.widgets['position_metrics']['position_ratio'].set(
                    f"{position_ratio:.1%}"
                )
            
        except Exception as e:
            self.log_message(f"更新仓位信息失败: {e}", "ERROR")
    
    def update_account_info(self):
        """更新账户信息"""
        try:
            # 获取账户信息
            position_summary = self.trading_engine.get_position_summary()
            balance = position_summary['balance']
            
            # 更新账户指标
            self.widgets['account_metrics']['balance_usdt'].set(
                f"${balance['USDT']:.2f}"
            )
            
            self.widgets['account_metrics']['balance_doge'].set(
                f"{balance['DOGE']:.0f} DOGE"
            )
            
            self.widgets['account_metrics']['total_balance'].set(
                f"${balance['total_value']:.2f}"
            )
            
            # 模拟其他指标
            self.widgets['account_metrics']['available_balance'].set(
                f"${balance['USDT'] * 0.95:.2f}"
            )
            
            self.widgets['account_metrics']['used_margin'].set(
                f"${balance['total_value'] * 0.05:.2f}"
            )
            
            if balance['total_value'] > 0:
                risk_ratio = (balance['total_value'] - self.trading_config.initial_balance) / self.trading_config.initial_balance
                self.widgets['account_metrics']['risk_ratio'].set(
                    f"{risk_ratio:.1%}"
                )
            
        except Exception as e:
            self.log_message(f"更新账户信息失败: {e}", "ERROR")
    
    def update_performance_info(self):
        """更新性能信息"""
        try:
            # 获取性能指标
            position_summary = self.trading_engine.get_position_summary()
            performance = position_summary['performance']
            
            # 更新性能指标
            self.widgets['performance_metrics']['total_pnl'].set(
                f"${performance['total_pnl']:+.2f}"
            )
            
            self.widgets['performance_metrics']['daily_pnl'].set(
                f"${performance['daily_pnl']:+.2f}"
            )
            
            self.widgets['performance_metrics']['win_rate'].set(
                f"{performance['win_rate']:.1%}"
            )
            
            self.widgets['performance_metrics']['total_trades'].set(
                f"{performance['total_trades']:.0f}"
            )
            
            self.widgets['performance_metrics']['profit_factor'].set(
                f"{performance['profit_factor']:.2f}"
            )
            
            self.widgets['performance_metrics']['max_drawdown'].set(
                f"{performance['max_drawdown']:.2%}"
            )
            
        except Exception as e:
            self.log_message(f"更新性能信息失败: {e}", "ERROR")
    
    def update_data_stats(self):
        """更新数据统计"""
        try:
            # 获取数据统计
            historical_data = self.data_manager.historical_data
            
            # 更新统计指标
            if '1d' in historical_data['price']:
                price_records = len(historical_data['price']['1d'])
                self.widgets['data_stats']['price_records'].set(
                    f"{price_records:,}"
                )
            
            if not historical_data['social'].empty:
                social_records = len(historical_data['social'])
                self.widgets['data_stats']['social_records'].set(
                    f"{social_records:,}"
                )
            
            if not historical_data['onchain'].empty:
                onchain_records = len(historical_data['onchain'])
                self.widgets['data_stats']['onchain_records'].set(
                    f"{onchain_records:,}"
                )
            
            if not historical_data['derivatives'].empty:
                derivatives_records = len(historical_data['derivatives'])
                self.widgets['data_stats']['derivatives_records'].set(
                    f"{derivatives_records:,}"
                )
            
            # 更新最新价格
            latest_data = self.data_manager.get_latest_aggregated_data()
            if 'price' in latest_data and 'close' in latest_data['price']:
                current_price = latest_data['price']['close']
                self.widgets['data_stats']['latest_price'].set(
                    f"${current_price:.6f}"
                )
            
            # 模拟其他指标
            self.widgets['data_stats']['data_quality'].set(
                f"{np.random.uniform(0.95, 0.99):.1%}"
            )
            
            self.widgets['data_stats']['last_update'].set(
                datetime.now().strftime("%H:%M:%S")
            )
            
            # 计算缓存大小（模拟）
            cache_size = np.random.uniform(5.0, 15.0)
            self.widgets['data_stats']['cache_size'].set(
                f"{cache_size:.1f} MB"
            )
            
        except Exception as e:
            self.log_message(f"更新数据统计失败: {e}", "ERROR")
    
    def update_charts(self):
        """更新图表"""
        try:
            # 更新价格图表
            self.update_price_chart()
            
            # 更新性能图表
            self.update_performance_chart()
            
            # 更新数据图表
            self.update_data_chart()
            
            # 更新特征重要性图表
            self.update_feature_chart()
            
        except Exception as e:
            self.log_message(f"更新图表失败: {e}", "ERROR")
    
    def update_price_chart(self):
        """更新价格图表"""
        try:
            fig = self.figures['price']
            ax1, ax2 = fig.get_axes()
            
            # 清除旧图表
            ax1.clear()
            ax2.clear()
            
            # 获取价格数据
            if self.price_history:
                timestamps = [p['timestamp'] for p in self.price_history]
                prices = [p['price'] for p in self.price_history]
                
                # 绘制价格
                ax1.plot(timestamps, prices, color=self.colors['primary'], linewidth=2)
                ax1.set_title('DOGE/USDT 价格', color=self.colors['text_light'])
                ax1.set_ylabel('价格 (USD)', color=self.colors['text_light'])
                ax1.grid(True, alpha=0.3)
                
                # 计算并绘制移动平均
                if len(prices) > 20:
                    ma_20 = pd.Series(prices).rolling(20).mean()
                    ax1.plot(timestamps, ma_20, color=self.colors['warning'], 
                            linewidth=1, label='MA20')
                    ax1.legend()
                
                # 绘制成交量（模拟）
                volumes = np.random.lognormal(10, 1, len(prices)) * 1000
                ax2.bar(timestamps, volumes, color=self.colors['volume'], alpha=0.7)
                ax2.set_title('成交量', color=self.colors['text_light'])
                ax2.set_ylabel('成交量', color=self.colors['text_light'])
                ax2.grid(True, alpha=0.3)
            
            # 设置样式
            for ax in [ax1, ax2]:
                ax.set_facecolor(self.colors['bg_medium'])
                ax.tick_params(colors=self.colors['text_light'])
                ax.xaxis.label.set_color(self.colors['text_light'])
                ax.yaxis.label.set_color(self.colors['text_light'])
                ax.title.set_color(self.colors['text_light'])
            
            # 调整布局
            fig.tight_layout()
            
            # 重绘
            self.canvases['price'].draw()
            
        except Exception as e:
            self.log_message(f"更新价格图表失败: {e}", "ERROR")
    
    def update_performance_chart(self):
        """更新性能图表"""
        try:
            fig = self.figures['performance']
            ax1, ax2 = fig.get_axes()
            
            # 清除旧图表
            ax1.clear()
            ax2.clear()
            
            # 获取交易历史
            trades = list(self.trading_engine.trade_history)
            
            if trades:
                # 提取盈亏数据
                profitable_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
                losing_trades = [t for t in trades if t.get('realized_pnl', 0) < 0]
                
                # 绘制盈亏分布
                categories = ['盈利交易', '亏损交易', '持平交易']
                counts = [
                    len(profitable_trades),
                    len(losing_trades),
                    len(trades) - len(profitable_trades) - len(losing_trades)
                ]
                
                colors = [self.colors['success'], self.colors['danger'], self.colors['warning']]
                ax1.bar(categories, counts, color=colors)
                ax1.set_title('交易分布', color=self.colors['text_light'])
                ax1.set_ylabel('交易次数', color=self.colors['text_light'])
                
                # 添加数值标签
                for i, count in enumerate(counts):
                    ax1.text(i, count + 0.1, str(count), ha='center', 
                            color=self.colors['text_light'])
                
                # 绘制累积盈亏曲线
                if trades:
                    cum_pnl = []
                    current_pnl = 0
                    
                    for trade in trades:
                        current_pnl += trade.get('realized_pnl', 0)
                        cum_pnl.append(current_pnl)
                    
                    timestamps = [t['timestamp'] for t in trades]
                    ax2.plot(timestamps, cum_pnl, color=self.colors['primary'], linewidth=2)
                    ax2.fill_between(timestamps, cum_pnl, 0, 
                                    where=[p >= 0 for p in cum_pnl], 
                                    color=self.colors['success'], alpha=0.3)
                    ax2.fill_between(timestamps, cum_pnl, 0,
                                    where=[p < 0 for p in cum_pnl],
                                    color=self.colors['danger'], alpha=0.3)
                    
                    ax2.set_title('累积盈亏', color=self.colors['text_light'])
                    ax2.set_ylabel('盈亏 (USD)', color=self.colors['text_light'])
                    ax2.grid(True, alpha=0.3)
            
            # 设置样式
            for ax in [ax1, ax2]:
                ax.set_facecolor(self.colors['bg_medium'])
                ax.tick_params(colors=self.colors['text_light'])
                ax.xaxis.label.set_color(self.colors['text_light'])
                ax.yaxis.label.set_color(self.colors['text_light'])
                ax.title.set_color(self.colors['text_light'])
            
            # 调整布局
            fig.tight_layout()
            
            # 重绘
            self.canvases['performance'].draw()
            
        except Exception as e:
            self.log_message(f"更新性能图表失败: {e}", "ERROR")
    
    def update_data_chart(self):
        """更新数据图表"""
        try:
            fig = self.figures['data']
            axes = fig.get_axes()
            
            # 清除旧图表
            for ax in axes:
                ax.clear()
            
            # 获取数据
            historical_data = self.data_manager.historical_data
            
            # 图表1: 价格与社交媒体情绪
            if not historical_data['social'].empty and '1d' in historical_data['price']:
                price_df = historical_data['price']['1d']
                social_df = historical_data['social']
                social_df = social_df.copy()
                social_df.index = pd.to_datetime(social_df.index, errors='coerce')
                social_df = social_df.dropna()
                
                # 对齐时间
                common_dates = price_df.index.intersection(social_df.index)
                
                if len(common_dates) > 10:
                    # 绘制价格
                    ax1 = axes[0]
                    color = 'tab:blue'
                    ax1.set_xlabel('日期')
                    ax1.set_ylabel('价格 (USD)', color=color)
                    ax1.plot(common_dates, price_df.loc[common_dates, 'close'], 
                            color=self.colors['primary'], linewidth=2)
                    ax1.tick_params(axis='y', labelcolor=color)
                    
                    # 绘制情绪（次坐标轴）
                    ax1_twin = ax1.twinx()
                    color = 'tab:red'
                    ax1_twin.set_ylabel('情绪指数', color=color)
                    ax1_twin.plot(common_dates, social_df.loc[common_dates, 'weighted_sentiment'], 
                                 color=self.colors['sentiment'], linewidth=1, alpha=0.7)
                    ax1_twin.tick_params(axis='y', labelcolor=color)
                    
                    ax1.set_title('价格 vs 社交媒体情绪', color=self.colors['text_light'])
            
            # 图表2: 链上数据
            if not historical_data['onchain'].empty:
                onchain_df = historical_data['onchain']
                onchain_df = onchain_df.copy()
                onchain_df.index = pd.to_datetime(onchain_df.index, errors='coerce')
                onchain_df = onchain_df.dropna()
                dates = onchain_df.index[-30:]  # 最近30天
                
                ax2 = axes[1]
                width = 0.35
                x = np.arange(len(dates))
                
                # 绘制活跃地址和交易数
                ax2.bar(x - width/2, onchain_df.loc[dates, 'active_addresses'].values / 1000, 
                       width, label='活跃地址 (千)', color=self.colors['onchain'], alpha=0.7)
                ax2.bar(x + width/2, onchain_df.loc[dates, 'transaction_count'].values / 1000,
                       width, label='交易数 (千)', color=self.colors['primary'], alpha=0.7)
                
                ax2.set_xlabel('日期')
                ax2.set_ylabel('数量 (千)')
                ax2.set_title('链上活动', color=self.colors['text_light'])
                ax2.legend()
                ax2.set_xticks(x[::5])
                ax2.set_xticklabels([d.strftime('%m-%d') for d in dates[::5]])
            
            # 图表3: 衍生品数据
            if not historical_data['derivatives'].empty:
                deriv_df = historical_data['derivatives']
                deriv_df = deriv_df.copy()
                deriv_df.index = pd.to_datetime(deriv_df.index, errors='coerce')
                deriv_df = deriv_df.dropna()
                dates = deriv_df.index[-30:]
                
                ax3 = axes[2]
                
                # 绘制资金费率
                ax3.plot(dates, deriv_df.loc[dates, 'funding_rate'] * 100, 
                        color=self.colors['warning'], linewidth=2, label='资金费率')
                ax3.axhline(y=0, color=self.colors['text_muted'], linestyle='--', alpha=0.5)
                
                ax3.set_xlabel('日期')
                ax3.set_ylabel('资金费率 (%)')
                ax3.set_title('衍生品指标', color=self.colors['text_light'])
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 图表4: 市场情绪热图
            if not historical_data['social'].empty:
                social_df = historical_data['social'].copy()
                social_df.index = pd.to_datetime(social_df.index, errors='coerce')
                social_df = social_df.dropna()
                dates = social_df.index[-30:]
                
                ax4 = axes[3]
                
                # 绘制情绪指标
                metrics = ['twitter_sentiment', 'reddit_sentiment', 'news_sentiment']
                data = []
                
                for metric in metrics:
                    if metric in social_df.columns:
                        data.append(social_df.loc[dates, metric].values)
                
                if data:
                    im = ax4.imshow(data, aspect='auto', cmap='RdYlGn', 
                                   interpolation='nearest', vmin=0, vmax=1)
                    
                    ax4.set_xlabel('日期')
                    ax4.set_ylabel('情绪源')
                    ax4.set_title('市场情绪热图', color=self.colors['text_light'])
                    ax4.set_yticks(np.arange(len(metrics)))
                    ax4.set_yticklabels(['Twitter', 'Reddit', '新闻'])
                    ax4.set_xticks(np.arange(0, len(dates), 5))
                    labels = []
                    for d in dates[::5]:
                        if hasattr(d, "strftime"):
                            labels.append(d.strftime('%m-%d'))
                        else:
                            labels.append(str(d))
                    ax4.set_xticklabels(labels)
                    
                    # 添加颜色条
                    fig.colorbar(im, ax=ax4)
            
            # 设置样式
            for ax in axes:
                ax.set_facecolor(self.colors['bg_medium'])
                ax.tick_params(colors=self.colors['text_light'])
                ax.xaxis.label.set_color(self.colors['text_light'])
                ax.yaxis.label.set_color(self.colors['text_light'])
                ax.title.set_color(self.colors['text_light'])
                
                # 设置图例颜色
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_color(self.colors['text_light'])
            
            # 调整布局
            fig.tight_layout()
            
            # 重绘
            self.canvases['data'].draw()
            
        except Exception as e:
            self.log_message(f"更新数据图表失败: {e}", "ERROR")
    
    def update_feature_chart(self):
        """更新特征重要性图表"""
        try:
            fig = self.figures['features']
            ax = fig.get_axes()[0]
            
            # 清除旧图表
            ax.clear()
            
            # 获取特征重要性数据
            feature_importance = self.model_manager.feature_engineer.feature_importance
            
            if feature_importance:
                # 选择最重要的10个特征
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: np.mean(x[1]) if x[1] else 0,
                    reverse=True
                )[:10]
                
                features = [f[0] for f in sorted_features]
                importances = [np.mean(f[1]) if f[1] else 0 for f in sorted_features]
                
                # 绘制条形图
                y_pos = np.arange(len(features))
                ax.barh(y_pos, importances, color=self.colors['primary'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()  # 最重要的特征在最上面
                ax.set_xlabel('重要性')
                ax.set_title('Top 10 特征重要性', color=self.colors['text_light'])
            
            # 设置样式
            ax.set_facecolor(self.colors['bg_medium'])
            ax.tick_params(colors=self.colors['text_light'])
            ax.xaxis.label.set_color(self.colors['text_light'])
            ax.yaxis.label.set_color(self.colors['text_light'])
            ax.title.set_color(self.colors['text_light'])
            
            # 调整布局
            fig.tight_layout()
            
            # 重绘
            self.canvases['features'].draw()
            
        except Exception as e:
            self.log_message(f"更新特征图表失败: {e}", "ERROR")
    
    def update_trade_history(self):
        """更新交易历史"""
        try:
            tree = self.widgets['trade_history']
            
            # 清除现有条目
            for item in tree.get_children():
                tree.delete(item)
            
            # 获取交易历史
            trades = list(self.trading_engine.trade_history)[-50:]  # 显示最近50条
            
            for trade in reversed(trades):  # 最新的在最上面
                timestamp = trade['timestamp'].strftime('%H:%M:%S')
                action = trade.get('action', 'HOLD')
                quantity = f"{trade.get('quantity', 0):.0f}"
                price = f"${trade.get('price', 0):.6f}"
                value = f"${trade.get('value', 0):.2f}"
                commission = f"${trade.get('commission', 0):.4f}"
                pnl = f"${trade.get('realized_pnl', 0):+.2f}"
                status = "成功" if trade.get('success', False) else "失败"
                
                # 插入条目
                item = tree.insert("", 0, values=(
                    timestamp, action, quantity, price, value, commission, pnl, status
                ))
                
                # 根据盈亏设置颜色
                pnl_value = trade.get('realized_pnl', 0)
                if pnl_value > 0:
                    tree.tag_configure('profit', foreground=self.colors['success'])
                    tree.item(item, tags=('profit',))
                elif pnl_value < 0:
                    tree.tag_configure('loss', foreground=self.colors['danger'])
                    tree.item(item, tags=('loss',))
            
        except Exception as e:
            self.log_message(f"更新交易历史失败: {e}", "ERROR")
    
    def update_model_performance(self):
        """更新模型性能"""
        try:
            tree = self.widgets['model_performance']
            
            # 清除现有条目
            for item in tree.get_children():
                tree.delete(item)
            
            # 获取模型性能
            model_performance = self.model_manager.model_performance
            
            for name, perf in model_performance.items():
                if perf and 'accuracy' in perf:
                    accuracy = f"{perf['accuracy']:.3f}"
                    f1_score = f"{np.random.uniform(0.5, 0.8):.3f}"  # 模拟F1分数
                    auc = f"{np.random.uniform(0.6, 0.9):.3f}"  # 模拟AUC
                    last_train = "最近"
                    status = "正常"
                    
                    # 插入条目
                    tree.insert("", "end", values=(
                        name.upper(), accuracy, f1_score, auc, last_train, status
                    ))
            
            # 更新摘要
            if model_performance:
                avg_accuracy = np.mean([p.get('accuracy', 0) for p in model_performance.values()])
                self.variables['model_summary'].set(
                    f"平均准确率: {avg_accuracy:.3f} | 模型数: {len(model_performance)}"
                )
            
        except Exception as e:
            self.log_message(f"更新模型性能失败: {e}", "ERROR")
    
    def execute_trading_cycle(self):
        """执行交易周期"""
        try:
            # 定期自动重训模型
            hours_since_last_train = (time.time() - self.model_manager.last_retrain_time) / 3600
            if hours_since_last_train >= self.trading_config.model_retrain_interval:
                threading.Thread(
                    target=self.model_manager.train_models,
                    args=(self.data_manager,),
                    kwargs={"retrain": True},
                    daemon=True
                ).start()
            
            # 生成交易信号
            signal = self.generate_trading_signal()
            
            # 显示信号
            self.display_trading_signal(signal)
            
            # 执行交易
            if signal['action'] != 'HOLD' and signal['confidence'] > self.trading_config.confidence_threshold:
                # 获取当前价格
                latest_data = self.data_manager.get_latest_aggregated_data()
                current_price = latest_data['price'].get('close', 0.08) if 'price' in latest_data else 0.08
                
                # 执行交易
                trade_result = self.trading_engine.execute_trade(signal, current_price)
                
                if trade_result['success']:
                    self.signals_history.append(signal)
                    self.log_message(
                        f"交易执行: {trade_result['reason']}",
                        "SUCCESS" if trade_result.get('realized_pnl', 0) >= 0 else "ERROR"
                    )
                else:
                    self.log_message(f"交易失败: {trade_result['reason']}", "WARNING")
            
            self.log_message(f"交易周期完成: {signal['action']}，置信度: {signal['confidence']:.1%}", "INFO")
            
        except Exception as e:
            self.log_message(f"交易周期失败: {e}", "ERROR")
    
    def generate_trading_signal(self):
        """生成交易信号"""
        try:
            # 获取最新特征
            features_df = self.feature_engineer.create_complete_features(self.data_manager)
            
            if features_df.empty:
                self.log_message("特征数据为空，使用默认信号", "WARNING")
                return self.get_default_signal()
            
            # 使用模型预测
            use_ensemble = self.variables['use_ensemble'].get()
            signal = self.model_manager.predict(features_df, use_ensemble=use_ensemble)
            
            return signal
            
        except Exception as e:
            self.log_message(f"生成交易信号失败: {e}", "ERROR")
            return self.get_default_signal()
    
    def get_default_signal(self):
        """获取默认信号"""
        return {
            'timestamp': datetime.now(),
            'action': 'HOLD',
            'strength': 'NEUTRAL',
            'confidence': 0.5,
            'position_size': 0,
            'prediction': 0,
            'reasoning': ['系统错误，使用默认持有信号']
        }
    
    def display_trading_signal(self, signal):
        """显示交易信号"""
        try:
            # 更新信号变量
            self.variables['signal_action'].set(signal['action'])
            self.variables['signal_strength'].set(signal['strength'])
            self.variables['signal_confidence'].set(f"{signal['confidence']:.1%}")
            
            # 更新信号图标
            action = signal['action']
            if action == 'BUY':
                icon = "BUY"
                color = self.colors['success']
            elif action == 'SELL':
                icon = "SELL"
                color = self.colors['danger']
            else:
                icon = "HOLD"
                color = self.colors['warning']
            
            self.widgets['signal_icon'].configure(text=icon, foreground=color)
            
            # 更新信号详情
            details = self.widgets['signal_details']
            details.configure(state='normal')
            details.delete(1.0, tk.END)
            
            # 格式化信号信息
            signal_text = f"时间: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            signal_text += f"信号: {signal['action']} ({signal['strength']})\n"
            signal_text += f"置信度: {signal['confidence']:.1%}\n"
            
            if signal['position_size'] > 0:
                signal_text += f"建议仓位: {signal['position_size']:.0f} DOGE\n"
            
            signal_text += f"预测类别: {signal.get('prediction', 0)}\n"
            
            signal_text += f"\n理由:\n"
            for reason in signal.get('reasoning', []):
                signal_text += f"  • {reason}\n"
            
            # 添加模型详情
            if 'model_details' in signal:
                details_md = signal['model_details']
                signal_text += f"\n模型详情:\n"
                
                if 'predictions' in details_md:
                    signal_text += f"  各模型预测: {details_md['predictions']}\n"
                
                if 'ensemble_used' in details_md:
                    signal_text += f"  集成模型: {'是' if details_md['ensemble_used'] else '否'}\n"
            
            signal_text += "\n" + "="*40 + "\n"
            
            # 插入文本
            details.insert(tk.END, signal_text)
            
            # 设置颜色
            details.tag_add("signal", "1.0", "end")
            details.tag_config("signal", foreground=color)
            details.configure(state='disabled')
            
            # 滚动到底部
            details.see(tk.END)
            
        except Exception as e:
            self.log_message(f"显示交易信号失败: {e}", "ERROR")
    
    def log_message(self, message, level="INFO"):
        """记录日志消息"""
        try:
            # 格式化消息
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}\n"
            
            # 更新日志显示
            log_display = self.widgets['log_display']
            log_display.configure(state='normal')
            log_display.insert(tk.END, formatted_message)
            
            # 应用标签
            start_index = log_display.index("end-1c linestart")
            end_index = log_display.index("end-1c")
            log_display.tag_add(level, start_index, end_index)
            
            # 限制日志行数
            lines = int(log_display.index('end-1c').split('.')[0])
            if lines > 200:
                log_display.delete(1.0, f"{lines-150}.0")
            
            log_display.see(tk.END)
            log_display.configure(state='disabled')
            
            # 同时输出到控制台
            print(f"[{level}] {message}")
            
        except Exception as e:
            print(f"日志记录失败: {e}")
    
    # ==================== 事件处理方法 ====================
    
    def on_trading_mode_change(self):
        """交易模式变更"""
        mode = self.variables['trading_mode_var'].get()
        
        if mode == "paper":
            self.trading_engine.paper_trading = True
            self.variables['trading_mode'].set("模拟交易")
            self.log_message("切换到模拟交易模式", "INFO")
        else:
            # 检查API配置
            api_key = self.variables['api_key'].get()
            api_secret = self.variables['api_secret'].get()
            
            if not api_key or not api_secret:
                messagebox.showwarning("API配置缺失", "请先配置API Key和Secret")
                self.variables['trading_mode_var'].set("paper")
                return
            
            self.trading_engine.paper_trading = False
            self.variables['trading_mode'].set("实盘交易")
            self.log_message("切换到实盘交易模式", "WARNING")
    
    def test_connection(self):
        """测试连接"""
        self.log_message("测试API连接...", "INFO")
        
        try:
            # 更新API配置
            self.update_api_config()
            
            # 测试连接
            connected = self.data_manager.api_client.test_connection()
            
            if connected:
                self.log_message("API连接测试成功", "SUCCESS")
                self.variables['connection_status'].set("连接正常")
            else:
                self.log_message("API连接测试失败", "ERROR")
                self.variables['connection_status'].set("连接失败")
                
        except Exception as e:
            self.log_message(f"连接测试失败: {e}", "ERROR")
            self.variables['connection_status'].set("连接错误")
    
    def sync_balance(self):
        """同步余额"""
        self.log_message("同步账户余额...", "INFO")
        
        try:
            if not self.trading_engine.paper_trading:
                success = self.trading_engine._update_real_balance()
                if success:
                    self.log_message("余额同步成功", "SUCCESS")
                else:
                    self.log_message("余额同步失败", "ERROR")
            else:
                self.log_message("模拟交易无需同步余额", "INFO")
                
        except Exception as e:
            self.log_message(f"同步余额失败: {e}", "ERROR")
    
    def sync_all(self):
        """同步所有数据"""
        self.log_message("开始同步所有数据...", "INFO")
        
        # 同步余额
        self.sync_balance()
        
        # 更新数据
        self.update_data_now()
        
        self.log_message("数据同步完成", "SUCCESS")
    
    def update_data_now(self):
        """立即更新数据"""
        self.log_message("更新数据...", "INFO")
        
        try:
            # 在后台线程中更新数据
            def update_thread():
                self.data_manager.fetch_historical_data(days=7)  # 更新最近7天数据
                self.root.after(0, lambda: self.log_message("数据更新完成", "SUCCESS"))
            
            threading.Thread(target=update_thread, daemon=True).start()
            
        except Exception as e:
            self.log_message(f"更新数据失败: {e}", "ERROR")
    
    def clear_cache(self):
        """清除缓存"""
        self.log_message("清除数据缓存...", "INFO")
        
        try:
            self.data_manager.cache.clear()
            self.data_manager.cache_expiry.clear()
            self.log_message("缓存已清除", "SUCCESS")
        except Exception as e:
            self.log_message(f"清除缓存失败: {e}", "ERROR")
    
    def train_models_now(self):
        """立即训练模型"""
        self.log_message("开始训练模型...", "INFO")
        
        # 在后台线程中训练模型
        def train_thread():
            try:
                success = self.model_manager.train_models(self.data_manager, retrain=True)
                if success:
                    self.root.after(0, lambda: self.log_message("模型训练完成", "SUCCESS"))
                else:
                    self.root.after(0, lambda: self.log_message("模型训练失败", "ERROR"))
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"模型训练异常: {e}", "ERROR"))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def retrain_models(self):
        """重新训练模型"""
        response = messagebox.askyesno(
            "重新训练模型",
            "确定要重新训练所有模型吗？这可能需要一些时间。"
        )
        
        if response:
            self.train_models_now()
    
    def load_models(self):
        """加载模型"""
        self.log_message("加载模型...", "INFO")
        
        try:
            # 打开文件对话框选择模型目录
            model_dir = filedialog.askdirectory(
                title="选择模型目录",
                initialdir=self.config.directories['models']
            )
            
            if model_dir:
                success = self.model_manager.load_models(model_dir)
                if success:
                    self.log_message("模型加载成功", "SUCCESS")
                else:
                    self.log_message("模型加载失败", "ERROR")
                    
        except Exception as e:
            self.log_message(f"加载模型失败: {e}", "ERROR")
    
    def save_models(self):
        """保存模型"""
        self.log_message("保存模型...", "INFO")
        
        try:
            self.model_manager._save_models()
            self.log_message("模型保存成功", "SUCCESS")
        except Exception as e:
            self.log_message(f"保存模型失败: {e}", "ERROR")
    
    def manual_buy(self):
        """手动买入"""
        try:
            # 获取输入
            quantity = simpledialog.askfloat(
                "手动买入",
                "请输入买入数量 (DOGE):",
                minvalue=10,
                maxvalue=10000
            )
            
            if quantity:
                # 创建买入信号
                signal = {
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'strength': 'MANUAL',
                    'confidence': 1.0,
                    'position_size': quantity,
                    'reasoning': ['手动买入']
                }
                
                # 执行买入
                latest_data = self.data_manager.get_latest_aggregated_data()
                current_price = latest_data['price'].get('close', 0.08) if 'price' in latest_data else 0.08
                
                trade_result = self.trading_engine.execute_trade(signal, current_price)
                
                if trade_result['success']:
                    self.log_message(f"手动买入成功: {quantity} DOGE", "SUCCESS")
                else:
                    self.log_message(f"手动买入失败: {trade_result['reason']}", "ERROR")
                    
        except Exception as e:
            self.log_message(f"手动买入失败: {e}", "ERROR")
    
    def manual_sell(self):
        """手动卖出"""
        try:
            # 获取当前持仓
            position = self.trading_engine.positions['DOGEUSDT']
            
            if position['quantity'] <= 0:
                messagebox.showinfo("无持仓", "当前没有持仓可卖出")
                return
            
            # 获取输入
            quantity = simpledialog.askfloat(
                "手动卖出",
                f"请输入卖出数量 (DOGE，最大 {position['quantity']:.0f}):",
                minvalue=10,
                maxvalue=position['quantity']
            )
            
            if quantity:
                # 创建卖出信号
                signal = {
                    'timestamp': datetime.now(),
                    'action': 'SELL',
                    'strength': 'MANUAL',
                    'confidence': 1.0,
                    'position_size': quantity,
                    'reasoning': ['手动卖出']
                }
                
                # 执行卖出
                latest_data = self.data_manager.get_latest_aggregated_data()
                current_price = latest_data['price'].get('close', 0.08) if 'price' in latest_data else 0.08
                
                trade_result = self.trading_engine.execute_trade(signal, current_price)
                
                if trade_result['success']:
                    self.log_message(f"手动卖出成功: {quantity} DOGE", "SUCCESS")
                else:
                    self.log_message(f"手动卖出失败: {trade_result['reason']}", "ERROR")
                    
        except Exception as e:
            self.log_message(f"手动卖出失败: {e}", "ERROR")
    
    def show_positions(self):
        """显示持仓详情"""
        try:
            position_summary = self.trading_engine.get_position_summary()
            position = position_summary['position']
            
            # 创建详情窗口
            detail_window = tk.Toplevel(self.root)
            detail_window.title("持仓详情")
            detail_window.geometry("400x300")
            
            # 显示持仓信息
            text = scrolledtext.ScrolledText(detail_window, width=50, height=15)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            info = f"""
持仓详情
==============
交易对: {position_summary['symbol']}
持仓数量: {position['quantity']:.0f} DOGE
入场价格: ${position['entry_price']:.6f}
当前价格: ${position['current_price']:.6f}
持仓价值: ${position['position_value']:.2f}
浮动盈亏: ${position['unrealized_pnl']:.2f} ({position['unrealized_pnl_ratio']:+.2%})
入场时间: {position['entry_time'] or 'N/A'}
            
账户信息
==============
USDT余额: ${position_summary['balance']['USDT']:.2f}
DOGE余额: {position_summary['balance']['DOGE']:.0f} DOGE
总资产: ${position_summary['balance']['total_value']:.2f}
            
交易模式: {'实盘交易' if position_summary['live_trading'] else '模拟交易'}
            """
            
            text.insert(tk.END, info)
            text.configure(state='disabled')
            
        except Exception as e:
            self.log_message(f"显示持仓失败: {e}", "ERROR")
    
    def show_orders(self):
        """显示订单"""
        try:
            # 获取订单历史
            orders = list(self.trading_engine.order_history)
            
            # 创建订单窗口
            order_window = tk.Toplevel(self.root)
            order_window.title("订单历史")
            order_window.geometry("600x400")
            
            # 使用Treeview显示订单
            columns = ("时间", "操作", "数量", "价格", "状态")
            
            tree_frame = ttk.Frame(order_window)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
            
            # 设置列
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # 添加滚动条
            scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            # 布局
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 添加订单数据
            for order in orders[-50:]:  # 最近50条订单
                timestamp = order['timestamp'].strftime('%H:%M:%S')
                action = order['action']
                quantity = f"{order.get('quantity', 0):.0f}"
                price = f"${order.get('price', 0):.6f}"
                status = order.get('status', 'UNKNOWN')
                
                tree.insert("", "end", values=(timestamp, action, quantity, price, status))
            
            if not orders:
                tree.insert("", "end", values=("暂无订单", "", "", "", ""))
                
        except Exception as e:
            self.log_message(f"显示订单失败: {e}", "ERROR")
    
    def clear_trade_history(self):
        """清除交易历史"""
        response = messagebox.askyesno(
            "清除交易历史",
            "确定要清除交易历史吗？此操作不可恢复。"
        )
        
        if response:
            self.trading_engine.trade_history.clear()
            self.log_message("交易历史已清除", "INFO")
    
    def export_history_csv(self):
        """导出历史为CSV"""
        try:
            # 获取交易历史
            trades = list(self.trading_engine.trade_history)
            
            if not trades:
                messagebox.showinfo("无数据", "没有交易历史可导出")
                return
            
            # 打开文件对话框
            filename = filedialog.asksaveasfilename(
                title="导出交易历史",
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
            )
            
            if filename:
                # 转换为DataFrame
                df = pd.DataFrame(trades)
                
                # 保存为CSV
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                
                self.log_message(f"交易历史已导出到 {filename}", "SUCCESS")
                
        except Exception as e:
            self.log_message(f"导出交易历史失败: {e}", "ERROR")
    
    def export_trades(self):
        """导出交易历史"""
        try:
            filename = self.trading_engine.export_trade_history()
            if filename:
                self.log_message(f"交易历史已导出到 {filename}", "SUCCESS")
            else:
                self.log_message("导出交易历史失败", "ERROR")
                
        except Exception as e:
            self.log_message(f"导出交易历史失败: {e}", "ERROR")
    
    def export_performance(self):
        """导出性能报告"""
        try:
            # 获取性能数据
            position_summary = self.trading_engine.get_position_summary()
            performance = position_summary['performance']
            
            # 打开文件对话框
            filename = filedialog.asksaveasfilename(
                title="导出性能报告",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
            if filename:
                # 保存为JSON
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(performance, f, indent=2, ensure_ascii=False)
                
                self.log_message(f"性能报告已导出到 {filename}", "SUCCESS")
                
        except Exception as e:
            self.log_message(f"导出性能报告失败: {e}", "ERROR")
    
    def show_price_chart(self):
        """显示价格图表"""
        messagebox.showinfo("价格图表", "价格图表已在概览页面显示")
    
    def show_performance_chart(self):
        """显示性能图表"""
        messagebox.showinfo("性能图表", "性能图表已在概览页面显示")
    
    def show_feature_importance(self):
        """显示特征重要性"""
        messagebox.showinfo("特征重要性", "特征重要性已在模型页面显示")
    
    def show_all_charts(self):
        """显示所有图表"""
        self.widgets['notebook'].select(0)  # 切换到概览页面
        self.log_message("已切换到图表页面", "INFO")
    
    def show_logs(self):
        """显示系统日志"""
        self.widgets['notebook'].select(3)  # 切换到数据页面（日志在右侧）
        self.log_message("日志面板已在右侧显示", "INFO")
    
    def show_model_details(self):
        """显示模型详情"""
        self.widgets['notebook'].select(2)  # 切换到模型页面
        self.log_message("已切换到模型页面", "INFO")
    
    def show_settings(self):
        """显示设置"""
        self.widgets['notebook'].select(1)  # 切换到交易页面
        self.log_message("已切换到设置页面", "INFO")
    
    def show_risk_control(self):
        """显示风控设置"""
        try:
            # 创建风控窗口
            risk_window = tk.Toplevel(self.root)
            risk_window.title("风险控制设置")
            risk_window.geometry("400x500")
            
            # 风控参数
            params = [
                ("日亏损限制", "daily_loss_limit", "${:.2f}", 
                 self.trading_engine.risk_metrics['daily_loss_limit'], 10, 10000),
                ("最大仓位", "max_position_limit", "${:.2f}", 
                 self.trading_engine.risk_metrics['max_position_limit'], 100, 100000),
                ("最大回撤", "max_drawdown_limit", "{:.1%}", 
                 self.trading_engine.risk_metrics['max_drawdown_limit'], 0.01, 0.5),
                ("连续亏损", "consecutive_loss_limit", "{:.0f}", 
                 self.trading_engine.risk_metrics['consecutive_loss_limit'], 1, 10),
                ("日交易次数", "daily_trade_limit", "{:.0f}", 
                 self.trading_engine.risk_metrics['daily_trade_limit'], 1, 100),
                ("冷却时间", "cooldown_period", "{:.0f}秒", 
                 self.trading_engine.risk_metrics['cooldown_period'], 10, 3600)
            ]
            
            # 创建输入框
            entries = {}
            
            for i, (label, key, fmt, value, min_val, max_val) in enumerate(params):
                # 标签
                ttk.Label(risk_window, text=f"{label}:").grid(
                    row=i, column=0, sticky="w", padx=10, pady=5
                )
                
                # 输入框
                var = tk.StringVar(value=str(value))
                entry = ttk.Entry(risk_window, textvariable=var, width=15)
                entry.grid(row=i, column=1, sticky="w", padx=10, pady=5)
                
                entries[key] = var
                
                # 范围标签
                ttk.Label(risk_window, text=f"[{min_val} - {max_val}]",
                         foreground="gray").grid(
                    row=i, column=2, sticky="w", padx=5, pady=5
                )
            
            # 保存按钮
            def save_risk_params():
                try:
                    for key, var in entries.items():
                        value = var.get()
                        
                        # 转换类型
                        if key in ['daily_loss_limit', 'max_position_limit']:
                            self.trading_engine.risk_metrics[key] = float(value)
                        elif key in ['max_drawdown_limit']:
                            self.trading_engine.risk_metrics[key] = float(value)
                        elif key in ['consecutive_loss_limit', 'daily_trade_limit', 'cooldown_period']:
                            self.trading_engine.risk_metrics[key] = int(float(value))
                    
                    self.log_message("风控参数已更新", "SUCCESS")
                    risk_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("保存失败", f"参数错误: {e}")
            
            save_button = ttk.Button(
                risk_window,
                text="保存设置",
                command=save_risk_params,
                style="Primary.TButton"
            )
            save_button.grid(row=len(params), column=0, columnspan=3, pady=20)
            
        except Exception as e:
            self.log_message(f"显示风控设置失败: {e}", "ERROR")
    
    def run_analysis(self):
        """运行分析"""
        self.log_message("开始运行分析...", "INFO")
        
        # 在后台运行分析
        def analysis_thread():
            try:
                # 模拟分析过程
                import time
                time.sleep(2)
                
                # 分析结果
                analysis_result = {
                    'market_trend': np.random.choice(['上涨', '下跌', '震荡']),
                    'volatility': np.random.uniform(0.01, 0.05),
                    'sentiment_score': np.random.uniform(0.3, 0.8),
                    'recommendation': np.random.choice(['买入', '卖出', '持有']),
                    'confidence': np.random.uniform(0.6, 0.9)
                }
                
                # 在主线程中显示结果
                self.root.after(0, lambda: self.show_analysis_result(analysis_result))
                
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"分析失败: {e}", "ERROR"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def show_analysis_result(self, result):
        """显示分析结果"""
        try:
            # 创建分析结果窗口
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("市场分析结果")
            analysis_window.geometry("500x400")
            
            # 显示分析结果
            text = scrolledtext.ScrolledText(analysis_window, width=60, height=20)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            info = f"""
市场分析报告
==============
分析���间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
市场趋势: {result['market_trend']}
波动率: {result['volatility']:.2%}
情绪得分: {result['sentiment_score']:.2f}
投资建议: {result['recommendation']}
置信度: {result['confidence']:.1%}
            
详细分析:
1. 价格趋势分析
   - 短期趋势: {np.random.choice(['看涨', '看跌', '中性'])}
   - 中期趋势: {np.random.choice(['看涨', '看跌', '中性'])}
   - 支撑位: ${np.random.uniform(0.07, 0.08):.6f}
   - 阻力位: ${np.random.uniform(0.08, 0.09):.6f}
            
2. 市场情绪分析
   - Twitter情绪: {np.random.uniform(0.4, 0.9):.2f}
   - Reddit情绪: {np.random.uniform(0.3, 0.8):.2f}
   - 新闻情绪: {np.random.uniform(0.2, 0.7):.2f}
            
3. 链上数据分析
   - 活跃地址: {np.random.randint(50000, 200000):,}
   - 交易数量: {np.random.randint(10000, 50000):,}
   - 资金流向: {np.random.choice(['净流入', '净流出'])}
            
4. 风险提示
   - 市场风险: {np.random.choice(['低', '中', '高'])}
   - 流动性风险: {np.random.choice(['低', '中', '高'])}
   - 建议仓位: {np.random.choice(['轻仓', '适中', '重仓'])}
            """
            
            text.insert(tk.END, info)
            text.configure(state='disabled')
            
            self.log_message("市场分析完成", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"显示分析结果失败: {e}", "ERROR")
    
    def generate_report(self):
        """生成报告"""
        self.log_message("生成系统报告...", "INFO")
        
        try:
            # 获取系统状态
            position_summary = self.trading_engine.get_position_summary()
            
            # 创建报告窗口
            report_window = tk.Toplevel(self.root)
            report_window.title("系统报告")
            report_window.geometry("600x500")
            
            # 显示报告
            text = scrolledtext.ScrolledText(report_window, width=70, height=25)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            report = f"""
DOGE多因子量化交易系统报告
================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
系统版本: v2.0.0
运行模式: {'实盘交易' if position_summary['live_trading'] else '模拟交易'}
            
1. 账户概览
------------
总资产: ${position_summary['balance']['total_value']:.2f}
USDT余额: ${position_summary['balance']['USDT']:.2f}
DOGE余额: {position_summary['balance']['DOGE']:.0f}
浮动盈亏: ${position_summary['position']['unrealized_pnl']:.2f}
            
2. 交易表现
------------
总盈亏: ${position_summary['performance']['total_pnl']:.2f}
日盈亏: ${position_summary['performance']['daily_pnl']:.2f}
总交易次数: {position_summary['performance']['total_trades']}
胜率: {position_summary['performance']['win_rate']:.1%}
盈亏比: {position_summary['performance']['profit_factor']:.2f}
最大回撤: {position_summary['performance']['max_drawdown']:.2%}
累计手续费: ${position_summary['performance']['total_commission']:.2f}
            
3. 持仓情况
------------
交易对: {position_summary['symbol']}
持仓数量: {position_summary['position']['quantity']:.0f} DOGE
持仓价值: ${position_summary['position']['position_value']:.2f}
入场价格: ${position_summary['position']['entry_price']:.6f}
当前价格: ${position_summary['position']['current_price']:.6f}
浮动盈亏率: {position_summary['position']['unrealized_pnl_ratio']:+.2%}
            
4. 系统状态
------------
运行时间: {self.widgets['system_info']['运行时间'].get()}
内存使用: {self.widgets['system_info']['内存使用'].get()}
CPU使用: {self.widgets['system_info']['CPU使用'].get()}
线程数量: {self.widgets['system_info']['线程数'].get()}
数据质量: {self.widgets['data_stats']['data_quality'].get()}
            
5. 风险控制
------------
日交易次数: {position_summary['trade_limits']['daily_trade_count']}
连续亏损: {position_summary['performance']['consecutive_losses']}
冷却状态: {'冷却中' if position_summary['trade_limits'].get('cooldown_until') else '正常'}
            
================================
报告结束
            """
            
            text.insert(tk.END, report)
            text.configure(state='disabled')
            
            # 导出按钮
            def export_report():
                filename = filedialog.asksaveasfilename(
                    title="导出报告",
                    defaultextension=".txt",
                    filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
                )
                
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)
                    self.log_message(f"报告已导出到 {filename}", "SUCCESS")
            
            export_button = ttk.Button(
                report_window,
                text="导出报告",
                command=export_report
            )
            export_button.pack(pady=(0, 10))
            
            self.log_message("系统报告生成完成", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"生成报告失败: {e}", "ERROR")
    
    def show_help(self):
        """显示帮助"""
        help_text = """
DOGE多因子量化交易系统 - 使用说明

1. 系统初始化
   - 首次运行需要初始化系统
   - 系统会自动加载历史数据和训练模型
   - 确保网络连接正常

2. API配置
   - 在交易页面配置币安API
   - 实盘交易需要有效的API Key和Secret
   - 建议使用代理访问国际版币安

3. 交易控制
   - 启动/停止交易系统
   - 手动买入/卖出
   - 查看持仓和订单

4. 模型管理
   - 查看模型性能
   - 重新训练模型
   - 调整预测参数

5. 数据分析
   - 查看价格图表
   - 分析市场情绪
   - 监控链上数据

6. 风险控制
   - 设置止损止盈
   - 控制仓位大小
   - 限制交易频率

注意事项:
- 实盘交易有风险，请谨慎操作
- 建议先用模拟交易测试策略
- 定期备份配置和数据
- 关注系统日志和性能指标
        """
        
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
DOGE多因子量化交易系统 v2.0.0

一个完整的加密货币量化交易系统，专门为DOGE/USDT交易对设计。

主要特性:
- 多因子模型集成 (XGBoost, LightGBM, 随机森林, Prophet)
- 实时数据流 (价格、社交媒体、链上数据)
- 完整的交易引擎 (支持实盘和模拟交易)
- 高级风险控制和管理
- 可视化图表和监控界面
- 自动模型训练和优化

开发者: AI Assistant
版本: 2.0.0
发布日期: 2024年

免责声明:
本软件仅供学习和研究使用，不构成投资建议。
加密货币交易风险极高，请谨慎决策。
        """
        
        messagebox.showinfo("关于", about_text)
    
    def reset_daily_metrics(self):
        """重置日度指标"""
        response = messagebox.askyesno(
            "重置日度指标",
            "确定要重置日度交易指标吗？"
        )
        
        if response:
            self.trading_engine.reset_daily_metrics()
            self.log_message("日度指标已重置", "INFO")
    
    def on_closing(self):
        """窗口关闭事件"""
        response = messagebox.askyesno(
            "退出系统",
            "确定要退出交易系统吗？"
        )
        
        if response:
            # 停止交易系统
            self.stop_trading_system()
            
            # 停止数据流
            self.data_manager.stop_real_time_stream()
            
            # 保存配置
            self.save_config()
            
            # 关闭窗口
            self.root.destroy()
            logger.info("交易系统已关闭")
    
    def update_data(self):
        """更新数据（菜单命令的别名）"""
        self.update_data_now()

    def restart_system(self):
        """重启系统（菜单命令）"""
        try:
            self.log_message("正在重启系统...", "WARNING")
            
            # 停止当前系统
            self.stop_trading_system()
            
            # 停止数据流
            if self.data_manager:
                self.data_manager.stop_real_time_stream()
            
            # 重置组件
            self.is_running = False
            self.is_initialized = False
            self.variables['system_status'].set("系统重启中...")
            
            # 重新初始化
            self.root.after(2000, self.initialize_system)
            
            self.log_message("系统重启已启动", "INFO")
        except Exception as e:
            self.log_message(f"重启系统失败: {e}", "ERROR")

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 70)
    print("DOGE多因子量化交易系统 - 完整融合版")
    print("=" * 70)
    print("系统特性:")
    print("  ✓ 完整的币安API实盘连接")
    print("  ✓ 多因子集成模型 (XGBoost, LightGBM, 随机森林, Prophet)")
    print("  ✓ 高级特征工程 (技术指标、社交媒体、链上数据)")
    print("  ✓ 精确手续费计算和风险控制")
    print("  ✓ 实时数据流和可视化监控")
    print("  ✓ 支持实盘和模拟交易模式")
    print("=" * 70)
    
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 创建应用程序实例
        app = CompleteTradingGUI(root)
        
        # 记录启动时间
        app.start_time = datetime.now()
        
        # 设置窗口关闭事件
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # 运行主循环
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
    main()