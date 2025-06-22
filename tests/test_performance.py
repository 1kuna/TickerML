#!/usr/bin/env python3
"""
Performance Testing Suite for TickerML
Tests latency and performance of real-time processing pipeline

This suite measures:
1. Model inference latency
2. Feature generation speed
3. Risk calculation performance
4. Database operation speed
5. Memory usage optimization
"""

import time
import psutil
import sqlite3
import numpy as np
import pandas as pd
import torch
import sys
import os
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pc.models.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from raspberry_pi.risk_manager import AdvancedRiskManager
from pc.enhanced_features import EnhancedFeatureGenerator

class PerformanceTester:
    """Main performance testing class"""
    
    def __init__(self):
        self.results = {}
        self.temp_db = None
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up temporary test environment"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
        
        # Create test data
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Create OHLCV table with sample data
        cursor.execute('''
            CREATE TABLE ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            )
        ''')
        
        # Insert test data (1000 records)
        test_data = []
        base_time = time.time() - 86400  # 24 hours ago
        
        for i in range(1000):
            timestamp = base_time + (i * 60)  # 1-minute intervals
            price = 50000 + np.random.normal(0, 500)  # BTC-like prices
            
            test_data.append((
                timestamp, 'BTCUSDT',
                price, price * 1.001, price * 0.999, price + np.random.normal(0, 100),
                np.random.exponential(1000)
            ))
        
        cursor.executemany('''
            INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', test_data)
        
        conn.commit()
        conn.close()
        print(f"Created test database: {self.temp_db}")
    
    def test_model_inference_speed(self):
        """Test 1: Decision Transformer inference latency"""
        print("\n🧠 Testing model inference speed...")
        
        # Create model
        config = DecisionTransformerConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=3,
            use_bf16=False  # CPU testing
        )
        model = DecisionTransformer(config)
        model.eval()
        
        # Test parameters
        batch_size = 1
        seq_len = 10
        feature_dim = 256
        num_iterations = 100
        
        # Create test inputs
        states = torch.randn(batch_size, seq_len, feature_dim)
        actions = torch.randint(0, 3, (batch_size, seq_len))
        returns_to_go = torch.randn(batch_size, seq_len, 1)
        timesteps = torch.arange(seq_len).unsqueeze(0)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(states, actions, returns_to_go, timesteps)
        
        # Measure inference times
        inference_times = []
        
        with torch.no_grad():
            for i in range(num_iterations):
                start_time = time.perf_counter()
                outputs = model(states, actions, returns_to_go, timesteps)
                end_time = time.perf_counter()
                
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        p50_time = statistics.median(inference_times)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)
        
        results = {
            'avg_latency_ms': avg_time,
            'p50_latency_ms': p50_time,
            'p95_latency_ms': p95_time,
            'p99_latency_ms': p99_time,
            'throughput_inferences_per_sec': 1000 / avg_time,
            'total_iterations': num_iterations
        }
        
        print(f"  ✅ Average latency: {avg_time:.2f}ms")
        print(f"  ✅ P95 latency: {p95_time:.2f}ms")
        print(f"  ✅ Throughput: {results['throughput_inferences_per_sec']:.1f} inferences/sec")
        
        self.results['model_inference'] = results
        return results
    
    def test_feature_generation_speed(self):
        """Test 2: Feature generation performance"""
        print("\n📊 Testing feature generation speed...")
        
        feature_generator = EnhancedFeatureGenerator(self.temp_db)
        
        # Test parameters
        num_iterations = 50
        symbols = ['BTCUSDT']
        
        generation_times = []
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            try:
                features = feature_generator.generate_technical_features(symbols[0])
                end_time = time.perf_counter()
                
                if len(features) > 0:
                    generation_times.append((end_time - start_time) * 1000)
            except Exception as e:
                print(f"    Warning: Feature generation failed: {e}")
                continue
        
        if generation_times:
            avg_time = statistics.mean(generation_times)
            p95_time = np.percentile(generation_times, 95)
            
            results = {
                'avg_generation_time_ms': avg_time,
                'p95_generation_time_ms': p95_time,
                'successful_generations': len(generation_times),
                'throughput_features_per_sec': 1000 / avg_time if avg_time > 0 else 0
            }
            
            print(f"  ✅ Average generation time: {avg_time:.2f}ms")
            print(f"  ✅ P95 generation time: {p95_time:.2f}ms")
            print(f"  ✅ Success rate: {len(generation_times)}/{num_iterations}")
        else:
            results = {'error': 'All feature generations failed'}
            print("  ❌ All feature generations failed")
        
        self.results['feature_generation'] = results
        return results
    
    def test_risk_calculation_speed(self):
        """Test 3: Risk management calculation performance"""
        print("\n⚠️ Testing risk calculation speed...")
        
        risk_manager = AdvancedRiskManager(db_path=self.temp_db)
        
        # Create test portfolio
        test_portfolio = {
            'BTCUSDT': {
                'quantity': 0.1,
                'market_value': 5000.0,
                'side': 'long',
                'entry_price': 50000.0,
                'current_price': 50000.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            },
            'total_value': 10000.0
        }
        
        symbols = ['BTCUSDT']
        num_iterations = 100
        
        risk_calc_times = []
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            try:
                risk_metrics = risk_manager.assess_portfolio_risk(test_portfolio, symbols)
                end_time = time.perf_counter()
                
                risk_calc_times.append((end_time - start_time) * 1000)
            except Exception as e:
                print(f"    Warning: Risk calculation failed: {e}")
                continue
        
        if risk_calc_times:
            avg_time = statistics.mean(risk_calc_times)
            p95_time = np.percentile(risk_calc_times, 95)
            
            results = {
                'avg_risk_calc_time_ms': avg_time,
                'p95_risk_calc_time_ms': p95_time,
                'successful_calculations': len(risk_calc_times),
                'throughput_calcs_per_sec': 1000 / avg_time if avg_time > 0 else 0
            }
            
            print(f"  ✅ Average risk calc time: {avg_time:.2f}ms")
            print(f"  ✅ P95 risk calc time: {p95_time:.2f}ms")
            print(f"  ✅ Success rate: {len(risk_calc_times)}/{num_iterations}")
        else:
            results = {'error': 'All risk calculations failed'}
            print("  ❌ All risk calculations failed")
        
        self.results['risk_calculation'] = results
        return results
    
    def test_database_performance(self):
        """Test 4: Database operation speed"""
        print("\n💾 Testing database performance...")
        
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Test read performance
        read_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            cursor.execute("SELECT * FROM ohlcv WHERE symbol = 'BTCUSDT' ORDER BY timestamp DESC LIMIT 100")
            results = cursor.fetchall()
            end_time = time.perf_counter()
            
            read_times.append((end_time - start_time) * 1000)
        
        # Test write performance
        write_times = []
        for i in range(50):
            test_record = (
                time.time() + i, 'TESTUSDT', 1000.0, 1001.0, 999.0, 1000.5, 100.0
            )
            
            start_time = time.perf_counter()
            cursor.execute('''
                INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', test_record)
            conn.commit()
            end_time = time.perf_counter()
            
            write_times.append((end_time - start_time) * 1000)
        
        conn.close()
        
        results = {
            'avg_read_time_ms': statistics.mean(read_times),
            'avg_write_time_ms': statistics.mean(write_times),
            'p95_read_time_ms': np.percentile(read_times, 95),
            'p95_write_time_ms': np.percentile(write_times, 95),
            'read_throughput_ops_per_sec': 1000 / statistics.mean(read_times),
            'write_throughput_ops_per_sec': 1000 / statistics.mean(write_times)
        }
        
        print(f"  ✅ Average read time: {results['avg_read_time_ms']:.2f}ms")
        print(f"  ✅ Average write time: {results['avg_write_time_ms']:.2f}ms")
        print(f"  ✅ Read throughput: {results['read_throughput_ops_per_sec']:.1f} ops/sec")
        print(f"  ✅ Write throughput: {results['write_throughput_ops_per_sec']:.1f} ops/sec")
        
        self.results['database'] = results
        return results
    
    def test_memory_usage(self):
        """Test 5: Memory usage analysis"""
        print("\n🧮 Testing memory usage...")
        
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model and measure memory increase
        config = DecisionTransformerConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=3
        )
        model = DecisionTransformer(config)
        
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test batch processing memory
        batch_sizes = [1, 4, 8, 16]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            # Create test batch
            seq_len = 10
            feature_dim = 256
            
            states = torch.randn(batch_size, seq_len, feature_dim)
            actions = torch.randint(0, 3, (batch_size, seq_len))
            returns_to_go = torch.randn(batch_size, seq_len, 1)
            timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
            
            # Measure memory during inference
            with torch.no_grad():
                outputs = model(states, actions, returns_to_go, timesteps)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage[f'batch_size_{batch_size}'] = current_memory
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'model_memory_mb': model_memory,
            'model_overhead_mb': model_memory - baseline_memory,
            'batch_memory_usage': memory_usage,
            'peak_memory_mb': max(memory_usage.values()),
            'memory_efficiency_mb_per_inference': (max(memory_usage.values()) - baseline_memory) / 16  # For batch size 16
        }
        
        print(f"  ✅ Baseline memory: {baseline_memory:.1f}MB")
        print(f"  ✅ Model overhead: {results['model_overhead_mb']:.1f}MB")
        print(f"  ✅ Peak memory: {results['peak_memory_mb']:.1f}MB")
        print(f"  ✅ Memory per inference: {results['memory_efficiency_mb_per_inference']:.1f}MB")
        
        self.results['memory'] = results
        return results
    
    def test_concurrent_performance(self):
        """Test 6: Concurrent processing performance"""
        print("\n🔄 Testing concurrent performance...")
        
        # Create simple inference function
        config = DecisionTransformerConfig(hidden_size=128, num_attention_heads=2, num_hidden_layers=2)
        model = DecisionTransformer(config)
        model.eval()
        
        def inference_task():
            states = torch.randn(1, 5, 256)
            actions = torch.randint(0, 3, (1, 5))
            returns_to_go = torch.randn(1, 5, 1)
            timesteps = torch.arange(5).unsqueeze(0)
            
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(states, actions, returns_to_go, timesteps)
            end_time = time.perf_counter()
            
            return (end_time - start_time) * 1000
        
        # Test different thread counts
        thread_counts = [1, 2, 4, 8]
        concurrent_results = {}
        
        for num_threads in thread_counts:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                start_time = time.perf_counter()
                
                # Submit 50 tasks
                futures = [executor.submit(inference_task) for _ in range(50)]
                
                # Wait for completion and collect results
                task_times = [future.result() for future in futures]
                
                end_time = time.perf_counter()
                total_time = (end_time - start_time) * 1000
            
            concurrent_results[f'threads_{num_threads}'] = {
                'total_time_ms': total_time,
                'avg_task_time_ms': statistics.mean(task_times),
                'throughput_tasks_per_sec': 50 / (total_time / 1000),
                'efficiency': (50 / num_threads) / (total_time / 1000)  # Tasks per thread per second
            }
            
            print(f"  ✅ {num_threads} threads: {total_time:.0f}ms total, {concurrent_results[f'threads_{num_threads}']['throughput_tasks_per_sec']:.1f} tasks/sec")
        
        self.results['concurrent'] = concurrent_results
        return concurrent_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📋 Performance Test Summary")
        print("=" * 50)
        
        # Calculate overall performance score
        scores = []
        
        # Model inference score (target: <50ms)
        if 'model_inference' in self.results:
            avg_latency = self.results['model_inference']['avg_latency_ms']
            inference_score = max(0, 100 - (avg_latency - 50) * 2) if avg_latency > 0 else 0
            scores.append(inference_score)
            print(f"Model Inference: {inference_score:.0f}/100 (avg: {avg_latency:.1f}ms)")
        
        # Database performance score (target: <10ms reads)
        if 'database' in self.results:
            read_time = self.results['database']['avg_read_time_ms']
            db_score = max(0, 100 - (read_time - 10) * 5) if read_time > 0 else 0
            scores.append(db_score)
            print(f"Database Performance: {db_score:.0f}/100 (reads: {read_time:.1f}ms)")
        
        # Memory efficiency score (target: <100MB overhead)
        if 'memory' in self.results:
            memory_overhead = self.results['memory']['model_overhead_mb']
            memory_score = max(0, 100 - memory_overhead) if memory_overhead > 0 else 100
            scores.append(memory_score)
            print(f"Memory Efficiency: {memory_score:.0f}/100 (overhead: {memory_overhead:.1f}MB)")
        
        # Overall score
        overall_score = statistics.mean(scores) if scores else 0
        
        print(f"\n🏆 Overall Performance Score: {overall_score:.0f}/100")
        
        if overall_score >= 80:
            print("🟢 EXCELLENT - Production ready performance")
        elif overall_score >= 60:
            print("🟡 GOOD - Acceptable performance with room for optimization")
        elif overall_score >= 40:
            print("🟠 FAIR - Performance issues may impact production")
        else:
            print("🔴 POOR - Significant performance optimization needed")
        
        return {
            'overall_score': overall_score,
            'individual_scores': {
                'inference': scores[0] if len(scores) > 0 else 0,
                'database': scores[1] if len(scores) > 1 else 0,
                'memory': scores[2] if len(scores) > 2 else 0
            },
            'detailed_results': self.results
        }
    
    def cleanup(self):
        """Clean up test environment"""
        if self.temp_db and os.path.exists(self.temp_db):
            os.remove(self.temp_db)
            print(f"Cleaned up test database: {self.temp_db}")
    
    def run_all_performance_tests(self):
        """Run complete performance test suite"""
        print("⚡ TickerML Performance Test Suite")
        print("=" * 40)
        
        try:
            # Run all tests
            self.test_model_inference_speed()
            self.test_feature_generation_speed()
            self.test_risk_calculation_speed()
            self.test_database_performance()
            self.test_memory_usage()
            self.test_concurrent_performance()
            
            # Generate final report
            report = self.generate_performance_report()
            
            return report
            
        except Exception as e:
            print(f"\n❌ Performance testing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            self.cleanup()

def main():
    """Main performance test execution"""
    tester = PerformanceTester()
    
    try:
        report = tester.run_all_performance_tests()
        return report is not None and report.get('overall_score', 0) >= 60
    except Exception as e:
        print(f"Performance testing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)