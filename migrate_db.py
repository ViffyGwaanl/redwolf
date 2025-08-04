#!/usr/bin/env python3
"""
数据库迁移脚本
用于将旧的dashscope平台配置迁移到custom_openai
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_db, PlatformConfig, ModelConfig
from sqlalchemy.orm import Session

def migrate_dashscope_to_custom_openai():
    """将dashscope平台配置迁移到custom_openai"""
    print("🔄 开始数据库迁移：dashscope -> custom_openai")
    
    # 获取数据库会话
    db_gen = get_db()
    db: Session = next(db_gen)
    
    try:
        # 1. 查找dashscope平台配置
        dashscope_config = db.query(PlatformConfig).filter(
            PlatformConfig.platform_type == "dashscope"
        ).first()
        
        if dashscope_config:
            print(f"📋 找到dashscope配置: enabled={dashscope_config.enabled}, has_api_key={bool(dashscope_config.api_key)}")
            
            # 检查是否已经有custom_openai配置
            existing_custom = db.query(PlatformConfig).filter(
                PlatformConfig.platform_type == "custom_openai"
            ).first()
            
            if existing_custom:
                print("⚠️ custom_openai配置已存在，删除旧的dashscope配置")
                db.delete(dashscope_config)
            else:
                print("🔄 迁移dashscope配置到custom_openai")
                # 更新平台类型
                dashscope_config.platform_type = "custom_openai"
                # 如果没有base_url，设置默认值
                if not dashscope_config.base_url:
                    dashscope_config.base_url = "https://api.openai.com"
        else:
            print("ℹ️ 未找到dashscope配置")
        
        # 2. 迁移dashscope模型配置
        dashscope_models = db.query(ModelConfig).filter(
            ModelConfig.platform_type == "dashscope"
        ).all()
        
        if dashscope_models:
            print(f"🔄 迁移 {len(dashscope_models)} 个dashscope模型配置")
            for model in dashscope_models:
                model.platform_type = "custom_openai"
        
        # 3. 提交更改
        db.commit()
        print("✅ 数据库迁移完成")
        
        # 4. 验证迁移结果
        custom_config = db.query(PlatformConfig).filter(
            PlatformConfig.platform_type == "custom_openai"
        ).first()
        
        if custom_config:
            print(f"✅ custom_openai配置验证成功: enabled={custom_config.enabled}")
        
        custom_models_count = db.query(ModelConfig).filter(
            ModelConfig.platform_type == "custom_openai"
        ).count()
        
        print(f"✅ custom_openai模型数量: {custom_models_count}")
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def clean_old_dashscope_records():
    """清理所有旧的dashscope记录"""
    print("🧹 清理旧的dashscope记录")
    
    db_gen = get_db()
    db: Session = next(db_gen)
    
    try:
        # 删除dashscope平台配置
        dashscope_configs = db.query(PlatformConfig).filter(
            PlatformConfig.platform_type == "dashscope"
        ).all()
        
        for config in dashscope_configs:
            db.delete(config)
            print(f"🗑️ 删除dashscope平台配置")
        
        # 删除dashscope模型配置
        dashscope_models = db.query(ModelConfig).filter(
            ModelConfig.platform_type == "dashscope"
        ).all()
        
        for model in dashscope_models:
            db.delete(model)
        
        if dashscope_models:
            print(f"🗑️ 删除 {len(dashscope_models)} 个dashscope模型配置")
        
        db.commit()
        print("✅ 清理完成")
        
    except Exception as e:
        print(f"❌ 清理失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def show_current_configs():
    """显示当前的平台配置"""
    print("📊 当前平台配置:")
    
    db_gen = get_db()
    db: Session = next(db_gen)
    
    try:
        configs = db.query(PlatformConfig).all()
        
        if not configs:
            print("   无平台配置")
        else:
            for config in configs:
                print(f"   {config.platform_type}: enabled={config.enabled}, "
                      f"has_api_key={bool(config.api_key)}, "
                      f"base_url={config.base_url or 'None'}")
        
        print("\n📊 当前模型配置数量:")
        platforms = ["custom_openai", "openrouter", "ollama", "lmstudio", "dashscope"]
        for platform in platforms:
            count = db.query(ModelConfig).filter(
                ModelConfig.platform_type == platform
            ).count()
            if count > 0:
                print(f"   {platform}: {count} 个模型")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("=== 数据库迁移工具 ===")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "migrate":
            migrate_dashscope_to_custom_openai()
        elif command == "clean":
            clean_old_dashscope_records()
        elif command == "show":
            show_current_configs()
        else:
            print("未知命令。可用命令: migrate, clean, show")
    else:
        print("使用方法:")
        print("  python migrate_db.py show     - 显示当前配置")
        print("  python migrate_db.py migrate  - 迁移dashscope到custom_openai")
        print("  python migrate_db.py clean    - 清理旧的dashscope记录")
        
        print("\n执行显示当前配置:")
        show_current_configs()