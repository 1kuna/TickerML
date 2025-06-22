#!/usr/bin/env python3
"""
System Health Validation Suite
Runs all existing tests and validates component health

This script:
1. Runs all existing test files
2. Validates component status
3. Checks system dependencies
4. Measures performance metrics
5. Generates health report
"""

import os
import sys
import subprocess
import time
import importlib
import sqlite3
import psutil
import torch
from pathlib import Path
from datetime import datetime
import json
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class SystemHealthValidator:
    """Main system health validation class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'components': {},
            'dependencies': {},
            'performance': {},
            'overall_status': 'unknown'
        }
    
    def check_dependencies(self):
        """Check all required dependencies"""
        print("\n🔍 Checking system dependencies...")
        
        dependencies = {
            'python_version': sys.version,
            'torch_available': False,
            'torch_version': None,
            'cuda_available': False,
            'kafka_running': False,
            'sqlite_available': False,
            'database_exists': False
        }
        
        # Check PyTorch
        try:
            import torch
            dependencies['torch_available'] = True
            dependencies['torch_version'] = torch.__version__
            dependencies['cuda_available'] = torch.cuda.is_available()
            print(f"  ✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        except ImportError:
            print("  ❌ PyTorch not available")
        
        # Check SQLite
        try:
            import sqlite3
            dependencies['sqlite_available'] = True
            
            # Check if main database exists
            db_path = self.project_root / "data" / "db" / "crypto_data.db"
            if db_path.exists():
                dependencies['database_exists'] = True
                print(f"  ✅ SQLite database exists: {db_path}")
            else:
                print(f"  ⚠️ SQLite database not found: {db_path}")
        except ImportError:
            print("  ❌ SQLite not available")
        
        # Check Kafka (look for running process)
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'kafka' in proc.info['name'].lower():
                    dependencies['kafka_running'] = True
                    break
            
            if dependencies['kafka_running']:
                print("  ✅ Kafka process detected")
            else:
                print("  ⚠️ Kafka process not detected")
        except Exception:
            print("  ❌ Error checking Kafka status")
        
        # Check other required packages
        required_packages = [
            'pandas', 'numpy', 'requests', 'websocket-client', 
            'pyyaml', 'flask', 'ta', 'asyncio'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} not available")
                dependencies[f'{package}_available'] = False
        
        self.results['dependencies'] = dependencies
        return dependencies
    
    def check_component_health(self):
        """Check health of major components"""
        print("\n🔧 Checking component health...")
        
        components = {}
        
        # Check exchange modules
        exchange_path = self.project_root / "raspberry_pi" / "exchanges"
        if exchange_path.exists():
            exchanges = ['binance.py', 'coinbase.py', 'kraken.py', 'kucoin.py']
            for exchange in exchanges:
                exchange_file = exchange_path / exchange
                components[f'exchange_{exchange[:-3]}'] = exchange_file.exists()
                status = "✅" if exchange_file.exists() else "❌"
                print(f"  {status} Exchange: {exchange[:-3]}")
        
        # Check Kafka components
        kafka_producers = self.project_root / "raspberry_pi" / "kafka_producers"
        kafka_consumers = self.project_root / "raspberry_pi" / "kafka_consumers"
        
        if kafka_producers.exists():
            producers = list(kafka_producers.glob("*.py"))
            components['kafka_producers_count'] = len(producers)
            print(f"  ✅ Kafka producers: {len(producers)} files")
        
        if kafka_consumers.exists():
            consumers = list(kafka_consumers.glob("*.py"))
            components['kafka_consumers_count'] = len(consumers)
            print(f"  ✅ Kafka consumers: {len(consumers)} files")
        
        # Check model files
        model_path = self.project_root / "pc" / "models"
        if model_path.exists():
            models = list(model_path.glob("*.py"))
            components['model_files_count'] = len(models)
            print(f"  ✅ Model files: {len(models)} files")
        
        # Check decision transformer specifically
        dt_path = self.project_root / "pc" / "models" / "decision_transformer.py"
        components['decision_transformer_exists'] = dt_path.exists()
        status = "✅" if dt_path.exists() else "❌"
        print(f"  {status} Decision Transformer")
        
        # Check paper trader
        paper_trader_path = self.project_root / "raspberry_pi" / "paper_trader.py"
        components['paper_trader_exists'] = paper_trader_path.exists()
        status = "✅" if paper_trader_path.exists() else "❌"
        print(f"  {status} Paper Trader")
        
        # Check risk manager
        risk_manager_path = self.project_root / "raspberry_pi" / "risk_manager.py"
        components['risk_manager_exists'] = risk_manager_path.exists()
        status = "✅" if risk_manager_path.exists() else "❌"
        print(f"  {status} Risk Manager")
        
        # Check arbitrage monitor
        arbitrage_path = self.project_root / "raspberry_pi" / "arbitrage_monitor.py"
        components['arbitrage_monitor_exists'] = arbitrage_path.exists()
        status = "✅" if arbitrage_path.exists() else "❌"
        print(f"  {status} Arbitrage Monitor")
        
        self.results['components'] = components
        return components
    
    def run_existing_tests(self):
        """Run all existing test files"""
        print("\n🧪 Running existing test suite...")
        
        test_dir = self.project_root / "tests"
        test_files = list(test_dir.glob("test_*.py"))
        test_results = {}
        
        # Skip integration test (we'll run it separately)
        test_files = [f for f in test_files if 'integration' not in f.name]
        
        total_tests = len(test_files)
        passed_tests = 0
        
        for test_file in test_files:
            test_name = test_file.stem
            print(f"  Running {test_name}...")
            
            try:
                # Run test using subprocess to isolate execution
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout per test
                )
                
                if result.returncode == 0:
                    test_results[test_name] = {
                        'status': 'passed',
                        'output': result.stdout,
                        'error': result.stderr
                    }
                    passed_tests += 1
                    print(f"    ✅ {test_name} PASSED")
                else:
                    test_results[test_name] = {
                        'status': 'failed',
                        'output': result.stdout,
                        'error': result.stderr,
                        'return_code': result.returncode
                    }
                    print(f"    ❌ {test_name} FAILED (code: {result.returncode})")
                    
            except subprocess.TimeoutExpired:
                test_results[test_name] = {
                    'status': 'timeout',
                    'error': 'Test timed out after 60 seconds'
                }
                print(f"    ⏰ {test_name} TIMEOUT")
                
            except Exception as e:
                test_results[test_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"    💥 {test_name} ERROR: {e}")
        
        test_summary = {
            'total': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'results': test_results
        }
        
        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} passed ({test_summary['success_rate']:.1%})")
        
        self.results['tests'] = test_summary
        return test_summary
    
    def measure_performance(self):
        """Measure system performance metrics"""
        print("\n⚡ Measuring performance metrics...")
        
        performance = {}
        
        # System metrics
        performance['cpu_count'] = psutil.cpu_count()
        performance['cpu_percent'] = psutil.cpu_percent(interval=1)
        performance['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        performance['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
        performance['memory_percent'] = psutil.virtual_memory().percent
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            performance['gpu_available'] = True
            performance['gpu_count'] = torch.cuda.device_count()
            performance['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            performance['gpu_available'] = False
        
        # Database size
        db_path = self.project_root / "data" / "db" / "crypto_data.db"
        if db_path.exists():
            performance['database_size_mb'] = db_path.stat().st_size / (1024**2)
            
            # Count records in main tables
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Count OHLCV records
                cursor.execute("SELECT COUNT(*) FROM ohlcv")
                performance['ohlcv_records'] = cursor.fetchone()[0]
                
                # Count recent records (last 24 hours)
                recent_cutoff = time.time() - 86400  # 24 hours
                cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE timestamp > ?", (recent_cutoff,))
                performance['recent_ohlcv_records'] = cursor.fetchone()[0]
                
                conn.close()
            except Exception as e:
                performance['database_error'] = str(e)
        
        # Model inference speed test (if Decision Transformer available)
        try:
            from pc.models.decision_transformer import DecisionTransformer, DecisionTransformerConfig
            
            config = DecisionTransformerConfig(
                hidden_size=256,
                num_attention_heads=4,
                num_hidden_layers=3,
                use_bf16=False
            )
            model = DecisionTransformer(config)
            
            # Test inference speed
            batch_size = 1
            seq_len = 10
            feature_dim = 256
            
            states = torch.randn(batch_size, seq_len, feature_dim)
            actions = torch.randint(0, 3, (batch_size, seq_len))
            returns_to_go = torch.randn(batch_size, seq_len, 1)
            timesteps = torch.arange(seq_len).unsqueeze(0)
            
            # Warmup
            with torch.no_grad():
                _ = model(states, actions, returns_to_go, timesteps)
            
            # Time inference
            start_time = time.time()
            num_inferences = 10
            
            with torch.no_grad():
                for _ in range(num_inferences):
                    _ = model(states, actions, returns_to_go, timesteps)
            
            total_time = time.time() - start_time
            performance['model_inference_ms'] = (total_time / num_inferences) * 1000
            
        except Exception as e:
            performance['model_inference_error'] = str(e)
        
        print(f"  ✅ CPU: {performance['cpu_count']} cores, {performance['cpu_percent']:.1f}% usage")
        print(f"  ✅ Memory: {performance['memory_available_gb']:.1f}GB available ({performance['memory_percent']:.1f}% used)")
        
        if performance.get('gpu_available'):
            print(f"  ✅ GPU: {performance['gpu_count']} devices, {performance['gpu_memory_gb']:.1f}GB memory")
        
        if 'database_size_mb' in performance:
            print(f"  ✅ Database: {performance['database_size_mb']:.1f}MB, {performance.get('ohlcv_records', 0):,} records")
        
        if 'model_inference_ms' in performance:
            print(f"  ✅ Model inference: {performance['model_inference_ms']:.1f}ms")
        
        self.results['performance'] = performance
        return performance
    
    def generate_health_report(self):
        """Generate overall health report"""
        print("\n📋 Generating health report...")
        
        # Calculate overall health score
        score = 0
        max_score = 0
        
        # Dependencies score (40% weight)
        dep_score = 0
        dep_max = 0
        for key, value in self.results['dependencies'].items():
            if key.endswith('_available') or key.endswith('_exists') or key == 'cuda_available':
                dep_max += 1
                if value:
                    dep_score += 1
            elif key == 'torch_available':
                dep_max += 2  # Higher weight for PyTorch
                if value:
                    dep_score += 2
        
        if dep_max > 0:
            score += (dep_score / dep_max) * 40
        max_score += 40
        
        # Components score (30% weight)
        comp_score = 0
        comp_max = 0
        for key, value in self.results['components'].items():
            if key.endswith('_exists'):
                comp_max += 1
                if value:
                    comp_score += 1
        
        if comp_max > 0:
            score += (comp_score / comp_max) * 30
        max_score += 30
        
        # Tests score (30% weight)
        if 'tests' in self.results and self.results['tests']['total'] > 0:
            test_success_rate = self.results['tests']['success_rate']
            score += test_success_rate * 30
        max_score += 30
        
        overall_score = score / max_score if max_score > 0 else 0
        
        # Determine overall status
        if overall_score >= 0.9:
            status = "excellent"
            emoji = "🟢"
        elif overall_score >= 0.7:
            status = "good"
            emoji = "🟡"
        elif overall_score >= 0.5:
            status = "fair"
            emoji = "🟠"
        else:
            status = "poor"
            emoji = "🔴"
        
        self.results['overall_status'] = status
        self.results['health_score'] = overall_score
        
        print(f"\n{emoji} Overall System Health: {status.upper()} ({overall_score:.1%})")
        
        # Print detailed breakdown
        print(f"\n📊 Health Breakdown:")
        dep_pct = (dep_score/dep_max) if dep_max > 0 else 0
        comp_pct = (comp_score/comp_max) if comp_max > 0 else 0
        print(f"  Dependencies: {dep_score}/{dep_max} ({dep_pct:.1%})")
        print(f"  Components: {comp_score}/{comp_max} ({comp_pct:.1%})")
        if 'tests' in self.results:
            print(f"  Tests: {self.results['tests']['passed']}/{self.results['tests']['total']} ({self.results['tests']['success_rate']:.1%})")
        
        return self.results
    
    def save_report(self, filename=None):
        """Save health report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_health_report_{timestamp}.json"
        
        report_path = self.project_root / "logs" / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {report_path}")
        return report_path
    
    def run_full_validation(self):
        """Run complete system health validation"""
        print("🔍 TickerML System Health Validation")
        print("=" * 50)
        
        try:
            # Run all validation steps
            self.check_dependencies()
            self.check_component_health()
            self.run_existing_tests()
            self.measure_performance()
            
            # Generate final report
            report = self.generate_health_report()
            
            # Save report
            self.save_report()
            
            print("\n✅ System health validation completed!")
            return report
            
        except Exception as e:
            print(f"\n❌ System health validation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    validator = SystemHealthValidator()
    report = validator.run_full_validation()
    
    # Return exit code based on health
    if report and report.get('health_score', 0) >= 0.7:
        return 0  # Success
    else:
        return 1  # Issues found

if __name__ == "__main__":
    exit(main())