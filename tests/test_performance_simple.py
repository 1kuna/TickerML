#!/usr/bin/env python3
"""
Simple Performance Testing Suite for TickerML
Tests basic system performance without complex dependencies

This suite measures:
1. Database operation speed
2. Feature generation performance
3. Risk calculation speed
4. Memory usage
5. Basic throughput
"""

import time
import psutil
import sqlite3
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import tempfile
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from raspberry_pi.risk_manager import AdvancedRiskManager

class SimplePerformanceTester:
    """Simplified performance testing class"""
    
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
    
    def test_database_performance(self):
        """Test database operation speed"""
        print("\n💾 Testing database performance...")
        
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Test read performance
        read_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            cursor.execute("SELECT * FROM ohlcv WHERE symbol = 'BTCUSDT' ORDER BY timestamp DESC LIMIT 100")
            results = cursor.fetchall()
            end_time = time.perf_counter()
            
            read_times.append((end_time - start_time) * 1000)
        
        # Test aggregation queries (more realistic)
        agg_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            cursor.execute('''
                SELECT symbol, AVG(close) as avg_price, COUNT(*) as count
                FROM ohlcv 
                WHERE timestamp > ? 
                GROUP BY symbol
            ''', (time.time() - 3600,))  # Last hour
            results = cursor.fetchall()
            end_time = time.perf_counter()
            
            agg_times.append((end_time - start_time) * 1000)
        
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
            'avg_agg_time_ms': statistics.mean(agg_times),
            'avg_write_time_ms': statistics.mean(write_times),
            'p95_read_time_ms': np.percentile(read_times, 95),
            'p95_agg_time_ms': np.percentile(agg_times, 95),
            'p95_write_time_ms': np.percentile(write_times, 95),
            'read_throughput_ops_per_sec': 1000 / statistics.mean(read_times),
            'write_throughput_ops_per_sec': 1000 / statistics.mean(write_times)
        }
        
        print(f"  ✅ Average read time: {results['avg_read_time_ms']:.2f}ms")
        print(f"  ✅ Average aggregation time: {results['avg_agg_time_ms']:.2f}ms")
        print(f"  ✅ Average write time: {results['avg_write_time_ms']:.2f}ms")
        print(f"  ✅ Read throughput: {results['read_throughput_ops_per_sec']:.1f} ops/sec")
        print(f"  ✅ Write throughput: {results['write_throughput_ops_per_sec']:.1f} ops/sec")
        
        self.results['database'] = results
        return results
    
    def test_risk_calculation_speed(self):
        """Test risk management calculation performance"""
        print("\n⚠️ Testing risk calculation speed...")
        
        risk_manager = AdvancedRiskManager(db_path=self.temp_db)
        
        # Create test portfolio scenarios
        test_portfolios = [
            # Single position
            {
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
            },
            # Multi-position portfolio
            {
                'BTCUSDT': {
                    'quantity': 0.05,
                    'market_value': 2500.0,
                    'side': 'long',
                    'entry_price': 50000.0,
                    'current_price': 50000.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                },
                'ETHUSDT': {
                    'quantity': 1.0,
                    'market_value': 3000.0,
                    'side': 'long',
                    'entry_price': 3000.0,
                    'current_price': 3000.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                },
                'total_value': 10000.0
            }
        ]
        
        all_risk_times = []
        
        for portfolio in test_portfolios:
            symbols = [k for k in portfolio.keys() if k != 'total_value']
            
            # Test risk assessment
            for _ in range(50):
                start_time = time.perf_counter()
                
                try:
                    risk_metrics = risk_manager.assess_portfolio_risk(portfolio, symbols)
                    end_time = time.perf_counter()
                    
                    all_risk_times.append((end_time - start_time) * 1000)
                except Exception as e:
                    print(f"    Warning: Risk calculation failed: {e}")
                    continue
        
        # Test position limit checking
        position_check_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            
            try:
                allowed, reason, max_size = risk_manager.check_position_limits(
                    'BTCUSDT', 2500.0, test_portfolios[0], ['BTCUSDT']
                )
                end_time = time.perf_counter()
                
                position_check_times.append((end_time - start_time) * 1000)
            except Exception as e:
                print(f"    Warning: Position check failed: {e}")
                continue
        
        if all_risk_times and position_check_times:
            results = {
                'avg_risk_assessment_ms': statistics.mean(all_risk_times),
                'avg_position_check_ms': statistics.mean(position_check_times),
                'p95_risk_assessment_ms': np.percentile(all_risk_times, 95),
                'p95_position_check_ms': np.percentile(position_check_times, 95),
                'risk_assessments_per_sec': 1000 / statistics.mean(all_risk_times),
                'position_checks_per_sec': 1000 / statistics.mean(position_check_times),
                'successful_risk_calcs': len(all_risk_times),
                'successful_position_checks': len(position_check_times)
            }
            
            print(f"  ✅ Average risk assessment: {results['avg_risk_assessment_ms']:.2f}ms")
            print(f"  ✅ Average position check: {results['avg_position_check_ms']:.2f}ms")
            print(f"  ✅ Risk assessments/sec: {results['risk_assessments_per_sec']:.1f}")
            print(f"  ✅ Position checks/sec: {results['position_checks_per_sec']:.1f}")
        else:
            results = {'error': 'Risk calculations failed'}
            print("  ❌ Risk calculations failed")
        
        self.results['risk_calculation'] = results
        return results
    
    def test_numpy_computation_speed(self):
        """Test NumPy computation performance (proxy for ML operations)"""
        print("\n🧮 Testing computation performance...")
        
        # Test matrix operations (common in ML)
        matrix_times = []
        for _ in range(100):
            # Create random matrices
            a = np.random.randn(256, 256)
            b = np.random.randn(256, 256)
            
            start_time = time.perf_counter()
            c = np.dot(a, b)
            end_time = time.perf_counter()
            
            matrix_times.append((end_time - start_time) * 1000)
        
        # Test statistical operations
        stats_times = []
        for _ in range(1000):
            data = np.random.randn(1000)
            
            start_time = time.perf_counter()
            mean = np.mean(data)
            std = np.std(data)
            corr = np.corrcoef(data[:-1], data[1:])[0, 1]  # Lag-1 correlation
            end_time = time.perf_counter()
            
            stats_times.append((end_time - start_time) * 1000)
        
        # Test array operations
        array_times = []
        for _ in range(1000):
            data = np.random.randn(10000)
            
            start_time = time.perf_counter()
            # Common financial calculations
            returns = np.diff(data) / data[:-1]
            cumulative = np.cumprod(1 + returns)
            rolling_mean = np.convolve(data, np.ones(20)/20, mode='valid')
            end_time = time.perf_counter()
            
            array_times.append((end_time - start_time) * 1000)
        
        results = {
            'avg_matrix_mult_ms': statistics.mean(matrix_times),
            'avg_stats_calc_ms': statistics.mean(stats_times),
            'avg_array_ops_ms': statistics.mean(array_times),
            'p95_matrix_mult_ms': np.percentile(matrix_times, 95),
            'p95_stats_calc_ms': np.percentile(stats_times, 95),
            'p95_array_ops_ms': np.percentile(array_times, 95),
            'matrix_ops_per_sec': 1000 / statistics.mean(matrix_times),
            'stats_ops_per_sec': 1000 / statistics.mean(stats_times),
            'array_ops_per_sec': 1000 / statistics.mean(array_times)
        }
        
        print(f"  ✅ Matrix multiplication: {results['avg_matrix_mult_ms']:.2f}ms")
        print(f"  ✅ Statistical calculations: {results['avg_stats_calc_ms']:.2f}ms")
        print(f"  ✅ Array operations: {results['avg_array_ops_ms']:.2f}ms")
        print(f"  ✅ Matrix ops/sec: {results['matrix_ops_per_sec']:.1f}")
        
        self.results['computation'] = results
        return results
    
    def test_memory_usage(self):
        """Test memory usage analysis"""
        print("\n🧮 Testing memory usage...")
        
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory usage with different data sizes
        memory_usage = {}
        
        # Small dataset (typical real-time processing)
        small_data = np.random.randn(1000, 50)  # 1000 records, 50 features
        small_memory = process.memory_info().rss / 1024 / 1024
        memory_usage['small_dataset'] = small_memory - baseline_memory
        
        # Medium dataset (batch processing)
        medium_data = np.random.randn(10000, 100)  # 10k records, 100 features
        medium_memory = process.memory_info().rss / 1024 / 1024
        memory_usage['medium_dataset'] = medium_memory - baseline_memory
        
        # Large dataset (historical analysis)
        large_data = np.random.randn(100000, 100)  # 100k records, 100 features
        large_memory = process.memory_info().rss / 1024 / 1024
        memory_usage['large_dataset'] = large_memory - baseline_memory
        
        # Clean up large arrays
        del small_data, medium_data, large_data
        
        # Test pandas DataFrame memory usage
        df = pd.DataFrame(np.random.randn(50000, 20))
        df_memory = process.memory_info().rss / 1024 / 1024
        memory_usage['dataframe_50k'] = df_memory - large_memory
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'memory_usage_by_dataset': memory_usage,
            'peak_memory_mb': large_memory,
            'total_memory_overhead_mb': large_memory - baseline_memory,
            'current_memory_mb': process.memory_info().rss / 1024 / 1024
        }
        
        print(f"  ✅ Baseline memory: {baseline_memory:.1f}MB")
        print(f"  ✅ Small dataset overhead: {memory_usage['small_dataset']:.1f}MB")
        print(f"  ✅ Medium dataset overhead: {memory_usage['medium_dataset']:.1f}MB")
        print(f"  ✅ Large dataset overhead: {memory_usage['large_dataset']:.1f}MB")
        print(f"  ✅ Peak memory: {large_memory:.1f}MB")
        
        self.results['memory'] = results
        return results
    
    def test_system_resources(self):
        """Test system resource utilization"""
        print("\n💻 Testing system resources...")
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # Disk I/O test
        start_io = psutil.disk_io_counters()
        
        # Perform some I/O operations
        with open(self.temp_db, 'rb') as f:
            data = f.read()
        
        time.sleep(0.1)  # Small delay
        
        end_io = psutil.disk_io_counters()
        
        # CPU usage test
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        results = {
            'cpu_count': cpu_count,
            'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
            'memory_total_gb': memory.total / 1024**3,
            'memory_available_gb': memory.available / 1024**3,
            'memory_percent_used': memory.percent,
            'cpu_usage_percent': cpu_percent,
            'avg_cpu_usage': statistics.mean(cpu_percent),
            'disk_read_mb': (end_io.read_bytes - start_io.read_bytes) / 1024**2 if start_io and end_io else 0,
            'disk_write_mb': (end_io.write_bytes - start_io.write_bytes) / 1024**2 if start_io and end_io else 0
        }
        
        print(f"  ✅ CPU cores: {cpu_count}")
        print(f"  ✅ CPU frequency: {results['cpu_frequency_mhz']:.0f}MHz")
        print(f"  ✅ Memory total: {results['memory_total_gb']:.1f}GB")
        print(f"  ✅ Memory available: {results['memory_available_gb']:.1f}GB")
        print(f"  ✅ Average CPU usage: {results['avg_cpu_usage']:.1f}%")
        
        self.results['system'] = results
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📋 Performance Test Summary")
        print("=" * 50)
        
        # Calculate performance scores
        scores = []
        
        # Database performance score (target: <10ms reads, <50ms writes)
        if 'database' in self.results:
            read_time = self.results['database']['avg_read_time_ms']
            write_time = self.results['database']['avg_write_time_ms']
            
            read_score = max(0, 100 - (read_time - 10) * 5) if read_time > 0 else 0
            write_score = max(0, 100 - (write_time - 50) * 2) if write_time > 0 else 0
            db_score = (read_score + write_score) / 2
            
            scores.append(db_score)
            print(f"Database Performance: {db_score:.0f}/100 (reads: {read_time:.1f}ms, writes: {write_time:.1f}ms)")
        
        # Risk calculation performance (target: <20ms)
        if 'risk_calculation' in self.results and 'avg_risk_assessment_ms' in self.results['risk_calculation']:
            risk_time = self.results['risk_calculation']['avg_risk_assessment_ms']
            risk_score = max(0, 100 - (risk_time - 20) * 3) if risk_time > 0 else 0
            scores.append(risk_score)
            print(f"Risk Calculation: {risk_score:.0f}/100 (avg: {risk_time:.1f}ms)")
        
        # Computation performance (target: <5ms for matrix ops)
        if 'computation' in self.results:
            matrix_time = self.results['computation']['avg_matrix_mult_ms']
            comp_score = max(0, 100 - (matrix_time - 5) * 10) if matrix_time > 0 else 0
            scores.append(comp_score)
            print(f"Computation Performance: {comp_score:.0f}/100 (matrix: {matrix_time:.1f}ms)")
        
        # Memory efficiency (target: <200MB overhead)
        if 'memory' in self.results:
            memory_overhead = self.results['memory']['total_memory_overhead_mb']
            memory_score = max(0, 100 - (memory_overhead - 200) * 0.5) if memory_overhead > 0 else 100
            scores.append(memory_score)
            print(f"Memory Efficiency: {memory_score:.0f}/100 (overhead: {memory_overhead:.1f}MB)")
        
        # System utilization
        if 'system' in self.results:
            cpu_usage = self.results['system']['avg_cpu_usage']
            memory_usage = self.results['system']['memory_percent_used']
            
            # Lower is better for utilization (more headroom)
            cpu_score = max(0, 100 - cpu_usage) if cpu_usage < 100 else 0
            mem_score = max(0, 100 - memory_usage) if memory_usage < 100 else 0
            system_score = (cpu_score + mem_score) / 2
            
            scores.append(system_score)
            print(f"System Utilization: {system_score:.0f}/100 (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)")
        
        # Overall performance score
        overall_score = statistics.mean(scores) if scores else 0
        
        print(f"\n🏆 Overall Performance Score: {overall_score:.0f}/100")
        
        # Performance classification
        if overall_score >= 80:
            status = "🟢 EXCELLENT - Production ready performance"
            recommendation = "System performs excellently across all metrics"
        elif overall_score >= 65:
            status = "🟡 GOOD - Acceptable performance with minor optimization opportunities"
            recommendation = "Consider optimizing the lowest scoring components"
        elif overall_score >= 45:
            status = "🟠 FAIR - Performance issues may impact production under load"
            recommendation = "Performance optimization recommended before production deployment"
        else:
            status = "🔴 POOR - Significant performance optimization required"
            recommendation = "Major performance improvements needed before production use"
        
        print(status)
        print(f"Recommendation: {recommendation}")
        
        # Specific recommendations
        print(f"\n📈 Performance Recommendations:")
        
        if 'database' in self.results:
            read_time = self.results['database']['avg_read_time_ms']
            if read_time > 20:
                print(f"  - Database reads are slow ({read_time:.1f}ms). Consider indexing optimization.")
        
        if 'risk_calculation' in self.results and 'avg_risk_assessment_ms' in self.results['risk_calculation']:
            risk_time = self.results['risk_calculation']['avg_risk_assessment_ms']
            if risk_time > 50:
                print(f"  - Risk calculations are slow ({risk_time:.1f}ms). Consider caching or optimization.")
        
        if 'memory' in self.results:
            memory_overhead = self.results['memory']['total_memory_overhead_mb']
            if memory_overhead > 500:
                print(f"  - High memory usage ({memory_overhead:.1f}MB). Consider memory optimization.")
        
        if 'system' in self.results:
            if self.results['system']['avg_cpu_usage'] > 80:
                print(f"  - High CPU usage. Consider process optimization or more CPU cores.")
            if self.results['system']['memory_percent_used'] > 85:
                print(f"  - High memory utilization. Consider adding more RAM.")
        
        return {
            'overall_score': overall_score,
            'status': status,
            'recommendation': recommendation,
            'individual_scores': {
                'database': scores[0] if len(scores) > 0 else 0,
                'risk_calculation': scores[1] if len(scores) > 1 else 0,
                'computation': scores[2] if len(scores) > 2 else 0,
                'memory': scores[3] if len(scores) > 3 else 0,
                'system': scores[4] if len(scores) > 4 else 0
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
        print("⚡ TickerML Performance Test Suite (Simplified)")
        print("=" * 50)
        
        try:
            # Run all tests
            self.test_database_performance()
            self.test_risk_calculation_speed()
            self.test_numpy_computation_speed()
            self.test_memory_usage()
            self.test_system_resources()
            
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
    tester = SimplePerformanceTester()
    
    try:
        report = tester.run_all_performance_tests()
        return report is not None and report.get('overall_score', 0) >= 50
    except Exception as e:
        print(f"Performance testing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)