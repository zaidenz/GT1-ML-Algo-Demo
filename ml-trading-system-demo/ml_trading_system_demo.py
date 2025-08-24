#!/usr/bin/env python3
"""
🤖 Advanced ML Trading System Demo
Professional algorithmic trading system powered by ensemble machine learning
and advanced market microstructure analysis.

Author: Zaiden
Created: 2025
License: Proprietary (Demo Version)

This demonstration showcases:
- Advanced feature engineering with 50+ market indicators
- Ensemble ML models (XGBoost, Random Forest)
- Time series cross-validation
- Production-ready signal generation
- Risk management integration

Note: Proprietary algorithms and specific trading logic are abstracted for IP protection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import yfinance as yf

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb

# Configure visualization
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MarketDataLoader:
    """Professional market data loading with robust error handling"""
    
    def __init__(self):
        print("📊 Market Data Loader Initialized")
    
    def load_data(self, ticker="QQQ", period_days=365):
        """
        Load market data with robust error handling and validation
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            print(f"📥 Loading {ticker} data ({period_days} days)...")
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]
            
            # Data validation
            df = df.dropna()
            
            print(f"✅ Data loaded: {len(df)} records")
            print(f"📅 Period: {df.index[0].date()} to {df.index[-1].date()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None

class AdvancedFeatureExtractor:
    """
    Professional-grade feature extraction for trading systems
    Implements market microstructure, technical, and behavioral features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        print("🔧 Advanced Feature Extractor Initialized")
        
    def calculate_rsi(self, prices, period=14):
        """Custom RSI implementation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def extract_features(self, df):
        """
        Extract comprehensive feature set for ML models
        """
        features = pd.DataFrame(index=df.index)
        print("⚙️ Extracting ML features...")
        
        # === PRICE ACTION FEATURES ===
        print("   📊 Price action analysis...")
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
            features[f'volatility_{period}'] = df['close'].rolling(period).std()
        
        # === MOMENTUM INDICATORS ===
        print("   ⚡ Momentum analysis...")
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'momentum_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # === VOLUME ANALYSIS ===
        print("   📈 Volume microstructure...")
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['price_volume'] = df['close'] * df['volume']
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # === RANGE ANALYSIS ===
        print("   📏 Range and positioning...")
        for period in [10, 20, 50]:
            features[f'high_{period}'] = df['high'].rolling(period).max()
            features[f'low_{period}'] = df['low'].rolling(period).min()
            features[f'range_{period}'] = features[f'high_{period}'] - features[f'low_{period}']
            features[f'position_in_range_{period}'] = (df['close'] - features[f'low_{period}']) / features[f'range_{period}']
        
        # === ADVANCED TECHNICAL INDICATORS ===
        print("   🎯 Advanced technical analysis...")
        features['rsi_14'] = self.calculate_rsi(df['close'], 14)
        features['rsi_7'] = self.calculate_rsi(df['close'], 7)
        
        # Bollinger Bands
        features['bb_middle'] = df['close'].rolling(20).mean()
        features['bb_std'] = df['close'].rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + (features['bb_std'] * 2)
        features['bb_lower'] = features['bb_middle'] - (features['bb_std'] * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # === MARKET TIMING FEATURES ===
        print("   ⏰ Temporal pattern analysis...")
        features['day_of_week'] = df.index.dayofweek
        features['hour'] = df.index.hour
        features['day_of_month'] = df.index.day
        
        # Cyclical encoding for time features
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # === PROPRIETARY FEATURES (ABSTRACTED) ===
        print("   🔒 Proprietary signal processing...")
        # Note: Actual proprietary features are abstracted for IP protection
        np.random.seed(42)  # For reproducible demo
        features['custom_signal_1'] = np.random.normal(0, 1, len(df))
        features['custom_signal_2'] = np.random.normal(0, 1, len(df))
        features['custom_timing'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
        
        # === TARGET VARIABLES ===
        print("   🎯 Target variable creation...")
        for horizon in [1, 5, 10, 20]:
            features[f'forward_return_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
            features[f'target_up_{horizon}'] = (features[f'forward_return_{horizon}'] > 0).astype(int)
        
        print(f"✅ Feature extraction complete: {len(features.columns)} features")
        return features.dropna()
    
    def prepare_ml_data(self, features_df, target_col='forward_return_5'):
        """
        Prepare features and targets for ML training
        """
        # Separate features from targets
        feature_cols = [col for col in features_df.columns 
                       if not col.startswith('forward_return') and not col.startswith('target_up')]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Remove NaN values
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y

class ProfessionalMLTrainer:
    """
    Enterprise-grade ML training pipeline for trading systems
    """
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        print("🤖 Professional ML Trainer Initialized")
        
    def train_ensemble(self, X, y, task_type='regression'):
        """
        Train ensemble of ML models with time series validation
        """
        print(f"\n🚀 Training {task_type} ensemble...")
        print("=" * 50)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Model configurations
        if task_type == 'regression':
            models_config = {
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
            }
        else:  # classification
            models_config = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
            }
        
        results = {}
        for name, model in models_config.items():
            print(f"📈 Training {name}...")
            
            try:
                # Cross-validation
                if task_type == 'regression':
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                    score = -cv_scores.mean()
                    metric = 'RMSE'
                else:
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                    score = cv_scores.mean()
                    metric = 'Accuracy'
                
                # Fit final model
                model.fit(X, y)
                self.models[f"{task_type}_{name}"] = model
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(X.columns, model.feature_importances_))
                    self.feature_importance[f"{task_type}_{name}"] = importance_dict
                
                results[name] = {
                    'score': score,
                    'std': cv_scores.std(),
                    'metric': metric
                }
                
                print(f"✅ {name}: {metric} = {score:.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"❌ {name}: Training failed - {str(e)}")
                results[name] = {'score': 0, 'std': 0, 'metric': 'Error'}
        
        return results
    
    def analyze_features(self, top_n=15):
        """
        Analyze feature importance across models
        """
        if not self.feature_importance:
            print("No feature importance data available")
            return []
        
        # Aggregate importance across models
        all_importance = {}
        for model_name, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(importance)
        
        # Average importance
        avg_importance = {
            feature: np.mean(importances)
            for feature, importances in all_importance.items()
        }
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 TOP {top_n} MOST IMPORTANT FEATURES:")
        print("=" * 60)
        for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
            print(f"{i:2d}. {feature:<25} {importance:.4f}")
        
        return sorted_features

class ProfessionalSignalGenerator:
    """
    Production-grade signal generation system
    """
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.models = trainer.models
        print("🎯 Professional Signal Generator Initialized")
    
    def generate_ensemble_predictions(self, X, task_type='regression'):
        """
        Generate ensemble predictions from trained models
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            if not model_name.startswith(task_type):
                continue
            
            name = model_name.replace(f"{task_type}_", "")
            
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                print(f"⚠️ Model {name} prediction failed: {e}")
        
        if not predictions:
            return np.zeros(len(X))
        
        # Simple ensemble average
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += pred / len(predictions)
        
        return ensemble_pred
    
    def generate_trading_signals(self, features_df, threshold=0.01):
        """
        Generate professional trading signals
        """
        print(f"🤖 Generating trading signals (threshold: {threshold})...")
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if not col.startswith('forward_return') and not col.startswith('target_up')]
        X = features_df[feature_cols]
        
        # Get predictions
        predictions = self.generate_ensemble_predictions(X, task_type='regression')
        
        # Create signals dataframe
        signals = pd.DataFrame(index=features_df.index)
        signals['ml_prediction'] = predictions
        signals['signal_strength'] = np.abs(predictions)
        
        # Generate trading signals with risk management
        signals['long_signal'] = (
            (predictions > threshold) &
            (features_df['volume_ratio'] > 1.2) &  # Volume confirmation
            (features_df['rsi_14'] < 80)  # Not overbought
        )
        
        signals['short_signal'] = (
            (predictions < -threshold) &
            (features_df['volume_ratio'] > 1.2) &  # Volume confirmation
            (features_df['rsi_14'] > 20)  # Not oversold
        )
        
        # Enhanced signals with proprietary filters (abstracted)
        signals['enhanced_long'] = (
            signals['long_signal'] &
            (features_df['custom_timing'] == 1)  # Proprietary timing (placeholder)
        )
        
        signals['enhanced_short'] = (
            signals['short_signal'] &
            (features_df['custom_timing'] == 1)  # Proprietary timing (placeholder)
        )
        
        print(f"✅ Generated {len(signals)} signals")
        return signals

def visualize_results(df, features_df, signals, important_features):
    """
    Create comprehensive visualizations of the trading system
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Price chart with signals
    axes[0,0].plot(df.index, df['close'], label='Price', linewidth=1)
    
    # Mark long signals
    long_signals = signals[signals['enhanced_long']]
    if len(long_signals) > 0:
        long_prices = [df.loc[idx, 'close'] for idx in long_signals.index if idx in df.index]
        axes[0,0].scatter(long_signals.index, long_prices, 
                         color='green', marker='^', s=50, label='Long Signals', alpha=0.7)
    
    # Mark short signals  
    short_signals = signals[signals['enhanced_short']]
    if len(short_signals) > 0:
        short_prices = [df.loc[idx, 'close'] for idx in short_signals.index if idx in df.index]
        axes[0,0].scatter(short_signals.index, short_prices,
                         color='red', marker='v', s=50, label='Short Signals', alpha=0.7)
    
    axes[0,0].set_title('🎯 ML Trading Signals', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Feature importance
    if important_features:
        top_features = important_features[:10]
        feature_names = [f[0] for f in top_features]
        feature_scores = [f[1] for f in top_features]
        
        axes[0,1].barh(range(len(feature_names)), feature_scores, color='steelblue')
        axes[0,1].set_yticks(range(len(feature_names)))
        axes[0,1].set_yticklabels(feature_names)
        axes[0,1].set_title('📊 Top ML Features')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Prediction distribution
    axes[1,0].hist(signals['ml_prediction'], bins=30, alpha=0.7, color='orange')
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('🎲 ML Prediction Distribution')
    axes[1,0].set_xlabel('Predicted Return')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Signal strength
    axes[1,1].plot(signals.index, signals['signal_strength'], alpha=0.7, color='purple')
    axes[1,1].set_title('💪 Signal Strength Over Time')
    axes[1,1].set_ylabel('Signal Strength')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution pipeline for the ML Trading System Demo
    """
    print("🚀 ML Trading System Demo Starting...")
    print("=" * 60)
    
    # 1. Load Data
    loader = MarketDataLoader()
    df = loader.load_data("QQQ", 500)  # NASDAQ ETF for demo
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    print(f"\n📈 Price Summary:")
    print(f"   Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Latest: ${df['close'].iloc[-1]:.2f}")
    
    # 2. Feature Engineering
    extractor = AdvancedFeatureExtractor()
    features_df = extractor.extract_features(df)
    
    print(f"\n📊 Feature Engineering Results:")
    print(f"   Samples: {len(features_df)}")
    print(f"   Features: {len(features_df.columns)}")
    print(f"   Date Range: {features_df.index[0].date()} to {features_df.index[-1].date()}")
    
    # 3. ML Training
    X, y = extractor.prepare_ml_data(features_df, target_col='forward_return_5')
    
    print(f"\n📋 Training Data Prepared:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Target Mean: {y.mean():.4f}")
    
    trainer = ProfessionalMLTrainer()
    reg_results = trainer.train_ensemble(X, y, task_type='regression')
    important_features = trainer.analyze_features(top_n=20)
    
    # 4. Signal Generation
    signal_gen = ProfessionalSignalGenerator(trainer)
    signals = signal_gen.generate_trading_signals(features_df, threshold=0.005)
    
    print(f"\n📊 SIGNAL STATISTICS:")
    print(f"   Total periods: {len(signals)}")
    print(f"   Long signals: {signals['long_signal'].sum()}")
    print(f"   Short signals: {signals['short_signal'].sum()}")
    print(f"   Enhanced long: {signals['enhanced_long'].sum()}")
    print(f"   Enhanced short: {signals['enhanced_short'].sum()}")
    
    # 5. Visualization
    print(f"\n📈 Generating visualizations...")
    visualize_results(df, features_df, signals, important_features)
    
    # 6. Performance Summary
    print(f"\n🎉 ML TRADING SYSTEM DEMO COMPLETE!")
    print("=" * 60)
    print(f"📊 System Performance Summary:")
    print(f"   Average prediction: {signals['ml_prediction'].mean():.4f}")
    print(f"   Prediction volatility: {signals['ml_prediction'].std():.4f}")
    print(f"   Signal rate: {(signals['enhanced_long'].sum() + signals['enhanced_short'].sum()) / len(signals) * 100:.1f}%")
    
    print(f"\n🔐 IP Protection Notice:")
    print("   • Proprietary timing algorithms abstracted")
    print("   • Custom feature engineering techniques protected")
    print("   • Model hyperparameters and ensemble weights secured")
    print("   • Specific trading logic anonymized")
    
    return {
        'data': df,
        'features': features_df,
        'signals': signals,
        'models': trainer.models,
        'feature_importance': important_features
    }

if __name__ == "__main__":
    # Run the complete ML trading system demo
    results = main()
