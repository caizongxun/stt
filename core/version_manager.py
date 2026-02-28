"""
Version Manager
版本管理器 - 管理所有策略版本
"""
from pathlib import Path
import importlib
import sys

class VersionManager:
    """
    策略版本管理器
    負責:
    1. 列出所有可用版本
    2. 動態加載指定版本
    3. 提供版本信息
    """
    
    def __init__(self):
        self.strategies_dir = Path('strategies')
        self.strategies_dir.mkdir(exist_ok=True)
    
    def list_versions(self) -> list:
        """
        列出所有可用的策略版本
        
        Returns:
            list: 版本列表,例如 ['v1', 'v2', 'v3']
        """
        if not self.strategies_dir.exists():
            return []
        
        versions = []
        for item in self.strategies_dir.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                # 檢查是否有__init__.py
                if (item / '__init__.py').exists():
                    versions.append(item.name)
        
        return sorted(versions)
    
    def get_version(self, version: str):
        """
        動態加載指定版本的模組
        
        Args:
            version: 版本名稱,例如 'v1'
        
        Returns:
            module: 版本模組
        """
        module_name = f"strategies.{version}"
        
        # 動態導入
        if module_name in sys.modules:
            return sys.modules[module_name]
        
        try:
            module = importlib.import_module(module_name)
            return module
        except ImportError as e:
            raise ImportError(f"無法加載版本 {version}: {e}")
    
    def get_version_info(self, version: str) -> dict:
        """
        獲取版本信息
        
        Args:
            version: 版本名稱
        
        Returns:
            dict: 版本信息
        """
        version_dir = self.strategies_dir / version
        info = {
            'name': version,
            'exists': version_dir.exists(),
            'has_trainer': (version_dir / 'trainer.py').exists(),
            'has_backtester': (version_dir / 'backtester.py').exists(),
            'has_config': (version_dir / 'config.py').exists(),
        }
        
        # 讀取README
        readme_path = version_dir / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                info['description'] = f.read()[:200]  # 前200字
        
        return info
