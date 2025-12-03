"""
CSDI模型单元测试
测试main_model.py中的核心功能，包括SpatialGCN、CSDI_base及其子类
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_model import SpatialGCN, CSDI_base, CSDI_PM25, CSDI_Physio, CSDI_Forecasting


class TestSpatialGCN:
    """测试SpatialGCN类的功能"""
    
    @pytest.fixture
    def sample_config(self):
        """创建测试用的配置"""
        return {
            'in_channels': 4,
            'hidden_channels': 8,
            'out_channels': 6,
            'num_layers': 2,
            'num_nodes': 3
        }
    
    @pytest.fixture
    def edge_index(self):
        """创建测试用的边索引"""
        # 创建一个简单的3节点图的边索引
        return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=torch.long)
    
    @pytest.fixture
    def edge_weight(self):
        """创建测试用的边权重"""
        return torch.ones(4, dtype=torch.float)
    
    @pytest.fixture
    def spatial_gcn(self, sample_config, edge_index, edge_weight):
        """创建SpatialGCN实例"""
        return SpatialGCN(
            in_channels=sample_config['in_channels'],
            hidden_channels=sample_config['hidden_channels'],
            out_channels=sample_config['out_channels'],
            num_layers=sample_config['num_layers'],
            edge_index=edge_index,
            edge_weight=edge_weight
        )
    
    def test_spatial_gcn_init(self, spatial_gcn, sample_config):
        """测试SpatialGCN初始化"""
        assert spatial_gcn.num_layers == sample_config['num_layers']
        assert len(spatial_gcn.convs) == sample_config['num_layers']
        assert spatial_gcn.convs[0].in_channels == sample_config['in_channels']
        assert spatial_gcn.convs[-1].out_channels == sample_config['out_channels']
    
    def test_spatial_gcn_forward(self, spatial_gcn):
        """测试SpatialGCN前向传播"""
        B, L, N, F = 2, 5, 3, 4  # batch, length, nodes, features
        x = torch.randn(B * L, N, F)
        
        output = spatial_gcn(x)
        
        # 检查输出形状
        assert output.shape == (B * L, N, spatial_gcn.convs[-1].out_channels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_spatial_gcn_single_layer(self, edge_index):
        """测试单层GCN"""
        gcn = SpatialGCN(
            in_channels=4,
            hidden_channels=8,
            out_channels=6,
            num_layers=1,
            edge_index=edge_index
        )
        
        B, L, N, F = 2, 5, 3, 4
        x = torch.randn(B * L, N, F)
        
        output = gcn(x)
        assert output.shape == (B * L, N, 6)
    
    def test_spatial_gcn_no_edge_weight(self, edge_index):
        """测试没有边权重的情况"""
        gcn = SpatialGCN(
            in_channels=4,
            hidden_channels=8,
            out_channels=6,
            num_layers=2,
            edge_index=edge_index,
            edge_weight=None
        )
        
        # 检查边权重是否被正确初始化为全1
        assert torch.allclose(gcn.edge_weight, torch.ones(edge_index.size(1)))


class TestCSDIBase:
    """测试CSDI_base类的功能"""
    
    @pytest.fixture
    def base_config(self):
        """创建基础配置"""
        return {
            "model": {
                "timeemb": 64,
                "featureemb": 16,
                "is_unconditional": False,
                "target_strategy": "mix",
                "num_sample_features": 10
            },
            "diffusion": {
                "num_steps": 10,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear"
            },
            "graph": {
                "num_stations": 3,
                "gcn_in": 4,
                "hidden": 8,
                "gcn_out": 4,
                "gcn_layers": 2
            }
        }
    
    @pytest.fixture
    def edge_index(self):
        """创建测试用的边索引"""
        return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=torch.long)
    
    @pytest.fixture
    def edge_weight(self):
        """创建测试用的边权重"""
        return torch.ones(4, dtype=torch.float)
    
    @pytest.fixture
    def csdi_base(self, base_config, edge_index, edge_weight):
        """创建CSDI_base实例"""
        device = torch.device('cpu')
        target_dim = 4
        
        # Mock diff_CSDI to avoid import issues
        with patch('main_model.diff_CSDI') as mock_diff:
            mock_model = Mock()
            mock_diff.return_value = mock_model
            
            csdi = CSDI_base(target_dim, base_config, device, edge_index, edge_weight)
            return csdi
    
    def test_csdi_base_init(self, csdi_base, base_config):
        """测试CSDI_base初始化"""
        assert csdi_base.target_dim == 4
        assert csdi_base.emb_time_dim == base_config["model"]["timeemb"]
        assert csdi_base.emb_feature_dim == base_config["model"]["featureemb"]
        assert csdi_base.is_unconditional == base_config["model"]["is_unconditional"]
        assert csdi_base.num_steps == base_config["diffusion"]["num_steps"]
    
    def test_time_embedding(self, csdi_base):
        """测试时间嵌入功能"""
        B, L = 2, 10
        pos = torch.randn(B, L)
        
        emb = csdi_base.time_embedding(pos, d_model=64)
        
        assert emb.shape == (B, L, 64)
        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()
    
    def test_time_embedding_different_dims(self, csdi_base):
        """测试不同维度的时间嵌入"""
        B, L, d_model = 3, 15, 128
        pos = torch.randn(B, L)
        
        emb = csdi_base.time_embedding(pos, d_model)
        
        assert emb.shape == (B, L, d_model)
    
    def test_get_randmask(self, csdi_base):
        """测试随机掩码生成"""
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        
        # 设置一些位置为0（原始缺失）
        observed_mask[0, 0, :3] = 0
        observed_mask[1, 1, 5:] = 0
        
        cond_mask = csdi_base.get_randmask(observed_mask)
        
        # 检查形状
        assert cond_mask.shape == observed_mask.shape
        
        # 检查原始缺失位置保持为0
        assert (cond_mask[0, 0, :3] == 0).all()
        assert (cond_mask[1, 1, 5:] == 0).all()
        
        # 检查条件掩码值在[0,1]范围内
        assert ((cond_mask >= 0) & (cond_mask <= 1)).all()
    
    def test_get_randmask_all_missing(self, csdi_base):
        """测试全缺失情况的随机掩码"""
        B, K, L = 2, 3, 5
        observed_mask = torch.zeros(B, K, L)
        
        cond_mask = csdi_base.get_randmask(observed_mask)
        
        # 全缺失时应该返回全0
        assert (cond_mask == 0).all()
    
    def test_get_hist_mask(self, csdi_base):
        """测试历史掩码生成"""
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        for_pattern_mask = torch.ones(B, K, L)
        
        cond_mask = csdi_base.get_hist_mask(observed_mask, for_pattern_mask)
        
        assert cond_mask.shape == observed_mask.shape
        assert ((cond_mask >= 0) & (cond_mask <= 1)).all()
    
    def test_get_hist_mask_mix_strategy(self, csdi_base):
        """测试混合策略的历史掩码"""
        csdi_base.target_strategy = "mix"
        
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        for_pattern_mask = torch.ones(B, K, L)
        
        with patch.object(csdi_base, 'get_randmask') as mock_randmask:
            mock_randmask.return_value = torch.ones(B, K, L) * 0.5
            
            cond_mask = csdi_base.get_hist_mask(observed_mask, for_pattern_mask)
            
            # 应该调用get_randmask
            mock_randmask.assert_called_once()
    
    def test_make_mixed_impute_mask(self, csdi_base):
        """测试混合插补掩码生成"""
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        
        # 设置一些原始缺失
        observed_mask[0, 0, :2] = 0
        
        cond_mask = csdi_base.make_mixed_impute_mask(
            observed_mask,
            block_prob=0.5,
            max_block_len=3,
            missing_ratio_range=(0.2, 0.4)
        )
        
        assert cond_mask.shape == observed_mask.shape
        assert ((cond_mask >= 0) & (cond_mask <= 1)).all()
        
        # 原始缺失位置应该保持为0
        assert (cond_mask[0, 0, :2] == 0).all()
    
    def test_make_mixed_impute_mask_all_missing(self, csdi_base):
        """测试全缺失时的混合插补掩码"""
        B, K, L = 2, 3, 5
        observed_mask = torch.zeros(B, K, L)
        
        cond_mask = csdi_base.make_mixed_impute_mask(observed_mask)
        
        assert (cond_mask == 0).all()
    
    def test_make_curriculum_impute_mask(self, csdi_base):
        """测试课程式插补掩码"""
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        
        # 测试训练早期
        cond_mask_early = csdi_base.make_curriculum_impute_mask(
            observed_mask,
            global_step=0,
            total_steps=100
        )
        
        # 测试训练后期
        cond_mask_late = csdi_base.make_curriculum_impute_mask(
            observed_mask,
            global_step=90,
            total_steps=100
        )
        
        assert cond_mask_early.shape == observed_mask.shape
        assert cond_mask_late.shape == observed_mask.shape
        
        # 后期应该有更多的缺失（更低的cond_mask值）
        assert cond_mask_late.sum() <= cond_mask_early.sum()
    
    def test_lerp(self, csdi_base):
        """测试线性插值"""
        start = torch.tensor(0.0)
        end = torch.tensor(10.0)
        
        # alpha=0应该返回start
        result_0 = csdi_base.lerp(start, end, 0.0)
        assert torch.allclose(result_0, start)
        
        # alpha=1应该返回end
        result_1 = csdi_base.lerp(start, end, 1.0)
        assert torch.allclose(result_1, end)
        
        # alpha=0.5应该返回中间值
        result_05 = csdi_base.lerp(start, end, 0.5)
        assert torch.allclose(result_05, torch.tensor(5.0))
    
    def test_get_test_pattern_mask(self, csdi_base):
        """测试测试模式掩码"""
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        test_pattern_mask = torch.ones(B, K, L)
        
        # 设置一些测试模式为0
        test_pattern_mask[0, :2, :3] = 0
        
        result = csdi_base.get_test_pattern_mask(observed_mask, test_pattern_mask)
        
        assert result.shape == observed_mask.shape
        # 只有observed_mask和test_pattern_mask都为1的位置才为1
        assert (result[0, :2, :3] == 0).all()
        assert (result[0, 2:, 3:] == 1).all()
    
    def test_get_side_info(self, csdi_base):
        """测试侧信息生成"""
        B, K, L = 2, 4, 10
        observed_tp = torch.randn(B, L)
        cond_mask = torch.ones(B, K, L)
        
        side_info = csdi_base.get_side_info(observed_tp, cond_mask)
        
        # 检查形状：应该是(B, emb_total_dim(+1), K, L)
        expected_dim = csdi_base.emb_total_dim + (0 if csdi_base.is_unconditional else 1)
        assert side_info.shape == (B, expected_dim, K, L)
        assert not torch.isnan(side_info).any()
    
    def test_get_side_info_unconditional(self, base_config, edge_index, edge_weight):
        """测试无条件模式的侧信息"""
        base_config["model"]["is_unconditional"] = True
        
        with patch('main_model.diff_CSDI') as mock_diff:
            mock_model = Mock()
            mock_diff.return_value = mock_model
            
            device = torch.device('cpu')
            csdi = CSDI_base(4, base_config, device, edge_index, edge_weight)
            
            B, K, L = 2, 4, 10
            observed_tp = torch.randn(B, L)
            cond_mask = torch.ones(B, K, L)
            
            side_info = csdi.get_side_info(observed_tp, cond_mask)
            
            # 无条件模式应该没有额外的mask通道
            expected_dim = csdi.emb_total_dim
            assert side_info.shape == (B, expected_dim, K, L)
    
    def test_set_input_to_diffmodel_conditional(self, csdi_base):
        """测试条件模式下的diffmodel输入设置"""
        B, K, L = 2, 4, 10
        noisy_data = torch.randn(B, K, L)
        observed_data = torch.randn(B, K, L)
        cond_mask = torch.ones(B, K, L)
        
        total_input = csdi_base.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        
        # 条件模式应该返回(B, 2, K, L)
        assert total_input.shape == (B, 2, K, L)
    
    def test_set_input_to_diffmodel_unconditional(self, base_config, edge_index, edge_weight):
        """测试无条件模式下的diffmodel输入设置"""
        base_config["model"]["is_unconditional"] = True
        
        with patch('main_model.diff_CSDI') as mock_diff:
            mock_model = Mock()
            mock_diff.return_value = mock_model
            
            device = torch.device('cpu')
            csdi = CSDI_base(4, base_config, device, edge_index, edge_weight)
            
            B, K, L = 2, 4, 10
            noisy_data = torch.randn(B, K, L)
            observed_data = torch.randn(B, K, L)
            cond_mask = torch.ones(B, K, L)
            
            total_input = csdi.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
            
            # 无条件模式应该返回(B, 1, K, L)
            assert total_input.shape == (B, 1, K, L)
    
    def test_apply_gcn_with_mask_flat(self, csdi_base):
        """测试带掩码的GCN应用（扁平版本）"""
        B, N, F_raw, L = 2, 3, 4, 10
        K_flat = N * F_raw
        
        data_flat = torch.randn(B, K_flat, L)
        obs_mask_flat = torch.ones(B, K_flat, L)
        cond_mask_flat = torch.ones(B, K_flat, L)
        
        # 设置一些NaN值
        data_flat[0, 0, :3] = float('nan')
        
        result = csdi_base.apply_gcn_with_mask_flat(data_flat, obs_mask_flat, cond_mask_flat)
        
        # 应该返回(B, N, F_out, L)
        assert result.shape == (B, N, csdi_base.gcn_out_dim, L)
        assert not torch.isnan(result).any()
    
    def test_apply_gcn_with_mask_flat_with_zeros(self, csdi_base):
        """测试带零掩码的GCN应用"""
        B, N, F_raw, L = 2, 3, 4, 10
        K_flat = N * F_raw
        
        data_flat = torch.randn(B, K_flat, L)
        obs_mask_flat = torch.zeros(B, K_flat, L)  # 全部掩码
        cond_mask_flat = torch.zeros(B, K_flat, L)
        
        result = csdi_base.apply_gcn_with_mask_flat(data_flat, obs_mask_flat, cond_mask_flat)
        
        assert result.shape == (B, N, csdi_base.gcn_out_dim, L)
        # 结果应该接近0（因为输入被掩码了）
        assert torch.abs(result).max() < 1e-6


class TestCSDIPM25:
    """测试CSDI_PM25类"""
    
    @pytest.fixture
    def pm25_config(self):
        """创建PM25配置"""
        return {
            "model": {
                "timeemb": 64,
                "featureemb": 16,
                "is_unconditional": False,
                "target_strategy": "mix"
            },
            "diffusion": {
                "num_steps": 10,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear"
            },
            "graph": {
                "num_stations": 3,
                "gcn_in": 4,
                "hidden": 8,
                "gcn_out": 4,
                "gcn_layers": 2
            }
        }
    
    @pytest.fixture
    def edge_index(self):
        return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=torch.long)
    
    @pytest.fixture
    def edge_weight(self):
        return torch.ones(4, dtype=torch.float)
    
    @pytest.fixture
    def csdi_pm25(self, pm25_config, edge_index, edge_weight):
        """创建CSDI_PM25实例"""
        device = torch.device('cpu')
        
        with patch('main_model.diff_CSDI') as mock_diff:
            mock_model = Mock()
            mock_diff.return_value = mock_model
            
            return CSDI_PM25(pm25_config, device, target_dim=72)
    
    def test_pm25_init(self, csdi_pm25):
        """测试CSDI_PM25初始化"""
        assert csdi_pm25.target_dim == 72
    
    def test_pm25_process_data(self, csdi_pm25):
        """测试PM25数据处理"""
        batch = {
            "observed_data": torch.randn(2, 10, 72),  # (B, L, K)
            "observed_mask": torch.ones(2, 10, 72),
            "timepoints": torch.randn(2, 10),
            "gt_mask": torch.ones(2, 10, 72),
            "hist_mask": torch.ones(2, 10, 72),
            "cut_length": torch.tensor([5, 3]),
            "absolute_time": ["2023-01-01", "2023-01-02"]
        }
        
        result = csdi_pm25.process_data(batch)
        
        # 应该返回7个元素
        assert len(result) == 7
        
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length, absolute_time = result
        
        # 检查维度转换：(B, L, K) -> (B, K, L)
        assert observed_data.shape == (2, 72, 10)
        assert observed_mask.shape == (2, 72, 10)
        assert observed_tp.shape == (2, 10)
        assert gt_mask.shape == (2, 72, 10)
        assert for_pattern_mask.shape == (2, 72, 10)
        assert cut_length.shape == (2,)


class TestCSDIPhysio:
    """测试CSDI_Physio类"""
    
    @pytest.fixture
    def physio_config(self):
        """创建Physio配置"""
        return {
            "model": {
                "timeemb": 64,
                "featureemb": 16,
                "is_unconditional": False,
                "target_strategy": "mix"
            },
            "diffusion": {
                "num_steps": 10,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear"
            },
            "graph": {
                "num_stations": 3,
                "gcn_in": 4,
                "hidden": 8,
                "gcn_out": 4,
                "gcn_layers": 2
            }
        }
    
    @pytest.fixture
    def edge_index(self):
        return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=torch.long)
    
    @pytest.fixture
    def edge_weight(self):
        return torch.ones(4, dtype=torch.float)
    
    @pytest.fixture
    def csdi_physio(self, physio_config, edge_index, edge_weight):
        """创建CSDI_Physio实例"""
        device = torch.device('cpu')
        
        with patch('main_model.diff_CSDI') as mock_diff:
            mock_model = Mock()
            mock_diff.return_value = mock_model
            
            return CSDI_Physio(physio_config, device, target_dim=35)
    
    def test_physio_init(self, csdi_physio):
        """测试CSDI_Physio初始化"""
        assert csdi_physio.target_dim == 35
    
    def test_physio_process_data(self, csdi_physio):
        """测试Physio数据处理"""
        batch = {
            "observed_data": torch.randn(2, 10, 35),  # (B, L, K)
            "observed_mask": torch.ones(2, 10, 35),
            "timepoints": torch.randn(2, 10),
            "gt_mask": torch.ones(2, 10, 35),
            "absolute_time": ["2023-01-01", "2023-01-02"]
        }
        
        result = csdi_physio.process_data(batch)
        
        # 应该返回7个元素
        assert len(result) == 7
        
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length, absolute_time = result
        
        # 检查维度转换
        assert observed_data.shape == (2, 35, 10)
        assert observed_mask.shape == (2, 35, 10)
        assert observed_tp.shape == (2, 10)
        assert gt_mask.shape == (2, 35, 10)
        assert for_pattern_mask.shape == (2, 35, 10)
        assert cut_length.shape == (2,)
        # Physio的cut_length应该全为0
        assert (cut_length == 0).all()


class TestCSDIForecasting:
    """测试CSDI_Forecasting类"""
    
    @pytest.fixture
    def forecasting_config(self):
        """创建预测配置"""
        return {
            "model": {
                "timeemb": 64,
                "featureemb": 16,
                "is_unconditional": False,
                "target_strategy": "mix",
                "num_sample_features": 10
            },
            "diffusion": {
                "num_steps": 10,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear"
            },
            "graph": {
                "num_stations": 3,
                "gcn_in": 4,
                "hidden": 8,
                "gcn_out": 4,
                "gcn_layers": 2
            }
        }
    
    @pytest.fixture
    def edge_index(self):
        return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=torch.long)
    
    @pytest.fixture
    def edge_weight(self):
        return torch.ones(4, dtype=torch.float)
    
    @pytest.fixture
    def csdi_forecasting(self, forecasting_config, edge_index, edge_weight):
        """创建CSDI_Forecasting实例"""
        device = torch.device('cpu')
        target_dim = 12
        
        with patch('main_model.diff_CSDI') as mock_diff:
            mock_model = Mock()
            mock_diff.return_value = mock_model
            
            return CSDI_Forecasting(forecasting_config, device, target_dim, edge_index, edge_weight)
    
    def test_forecasting_init(self, csdi_forecasting, forecasting_config):
        """测试CSDI_Forecasting初始化"""
        assert csdi_forecasting.target_dim_base == 12
        assert csdi_forecasting.num_sample_features == forecasting_config["model"]["num_sample_features"]
    
    def test_forecasting_process_data(self, csdi_forecasting):
        """测试预测数据处理"""
        batch = {
            "observed_data": torch.randn(2, 10, 12),  # (B, L, K)
            "observed_mask": torch.ones(2, 10, 12),
            "timepoints": torch.randn(2, 10),
            "gt_mask": torch.ones(2, 10, 12),
            "cut_length": torch.tensor([5, 3]),
            "absolute_time": ["2023-01-01", "2023-01-02"]
        }
        
        result = csdi_forecasting.process_data(batch)
        
        # 应该返回7个元素
        assert len(result) == 7
        
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length, feature_id = result
        
        # 检查维度转换
        assert observed_data.shape == (2, 12, 10)
        assert observed_mask.shape == (2, 12, 10)
        assert observed_tp.shape == (2, 10)
        assert gt_mask.shape == (2, 12, 10)
        assert for_pattern_mask.shape == (2, 12, 10)
        assert cut_length.shape == (2,)
        assert feature_id.shape == (2, 12)
    
    def test_get_forecast_cond_mask_tail(self, csdi_forecasting):
        """测试预测尾部条件掩码"""
        B, K, L = 3, 4, 10
        observed_mask = torch.ones(B, K, L)
        horizon = 3
        
        cond_mask = csdi_forecasting.get_forecast_cond_mask_tail(observed_mask, horizon)
        
        assert cond_mask.shape == observed_mask.shape
        
        # 最后horizon步应该为0
        assert (cond_mask[..., -horizon:] == 0).all()
        
        # 前面的步应该保持为1
        assert (cond_mask[..., :-horizon] == 1).all()
    
    def test_get_forecast_cond_mask_tail_edge_cases(self, csdi_forecasting):
        """测试预测尾部条件掩码边界情况"""
        B, K, L = 2, 3, 5
        observed_mask = torch.ones(B, K, L)
        
        # horizon=1
        cond_mask_1 = csdi_forecasting.get_forecast_cond_mask_tail(observed_mask, 1)
        assert (cond_mask_1[..., -1:] == 0).all()
        assert (cond_mask_1[..., :-1] == 1).all()
        
        # horizon=L (全部为预测目标)
        cond_mask_all = csdi_forecasting.get_forecast_cond_mask_tail(observed_mask, L)
        assert (cond_mask_all == 0).all()
    
    def test_sample_features(self, csdi_forecasting):
        """测试特征采样"""
        B, K_base, L = 3, 12, 10
        observed_data = torch.randn(B, K_base, L)
        observed_mask = torch.ones(B, K_base, L)
        feature_id = torch.arange(K_base).unsqueeze(0).expand(B, -1)
        gt_mask = torch.ones(B, K_base, L)
        
        # 设置采样数量小于基础数量
        csdi_forecasting.num_sample_features = 8
        csdi_forecasting.target_dim = 8
        
        extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask = csdi_forecasting.sample_features(
            observed_data, observed_mask, feature_id, gt_mask
        )
        
        # 检查形状
        assert extracted_data.shape == (B, 8, L)
        assert extracted_mask.shape == (B, 8, L)
        assert extracted_feature_id.shape == (B, 8)
        assert extracted_gt_mask.shape == (B, 8, L)
        
        # 检查特征ID在有效范围内
        assert (extracted_feature_id >= 0).all()
        assert (extracted_feature_id < K_base).all()
    
    def test_get_side_info_with_feature_id(self, csdi_forecasting):
        """测试带特征ID的侧信息生成"""
        B, K, L = 2, 8, 10
        observed_tp = torch.randn(B, L)
        cond_mask = torch.ones(B, K, L)
        feature_id = torch.randint(0, 12, (B, K))  # 原始特征空间有12个特征
        
        # 设置target_dim为采样后的数量
        csdi_forecasting.target_dim = 8
        csdi_forecasting.target_dim_base = 12
        
        side_info = csdi_forecasting.get_side_info(observed_tp, cond_mask, feature_id)
        
        expected_dim = csdi_forecasting.emb_total_dim + (0 if csdi_forecasting.is_unconditional else 1)
        assert side_info.shape == (B, expected_dim, K, L)
        assert not torch.isnan(side_info).any()


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def integration_config(self):
        """集成测试配置"""
        return {
            "model": {
                "timeemb": 32,
                "featureemb": 8,
                "is_unconditional": False,
                "target_strategy": "mix"
            },
            "diffusion": {
                "num_steps": 5,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear"
            },
            "graph": {
                "num_stations": 2,
                "gcn_in": 3,
                "hidden": 4,
                "gcn_out": 3,
                "gcn_layers": 2
            }
        }
    
    @pytest.fixture
    def edge_index(self):
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    @pytest.fixture
    def edge_weight(self):
        return torch.ones(2, dtype=torch.float)
    
    def test_full_pipeline_mock(self, integration_config, edge_index, edge_weight):
        """测试完整流程（使用Mock）"""
        device = torch.device('cpu')
        
        with patch('main_model.diff_CSDI') as mock_diff:
            # Mock扩散模型
            mock_model = Mock()
            mock_model.return_value = torch.randn(2, 3, 10)  # 模拟预测输出
            mock_diff.return_value = mock_model
            
            # 创建模型
            csdi = CSDI_base(3, integration_config, device, edge_index, edge_weight)
            
            # 创建测试数据
            B, K_flat, L = 2, 6, 10  # 2 stations * 3 features = 6
            observed_data = torch.randn(B, K_flat, L)
            observed_mask = torch.ones(B, K_flat, L)
            observed_tp = torch.randn(B, L)
            
            # 生成条件掩码
            cond_mask = csdi.get_randmask(observed_mask)
            
            # 生成侧信息
            side_info = csdi.get_side_info(observed_tp, cond_mask)
            
            # 测试损失计算
            loss = csdi.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train=1)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # 标量
            assert not torch.isnan(loss)
    
    def test_beta_schedules(self, integration_config, edge_index, edge_weight):
        """测试不同的beta调度"""
        device = torch.device('cpu')
        
        schedules = ["linear", "quad", "cosine"]
        
        for schedule in schedules:
            config = integration_config.copy()
            config["diffusion"]["schedule"] = schedule
            
            with patch('main_model.diff_CSDI') as mock_diff:
                mock_model = Mock()
                mock_diff.return_value = mock_model
                
                csdi = CSDI_base(3, config, device, edge_index, edge_weight)
                
                # 检查beta数组
                assert len(csdi.beta) == config["diffusion"]["num_steps"]
                assert (csdi.beta > 0).all()
                assert (csdi.beta < 1).all()
                
                # 检查alpha相关数组
                assert len(csdi.alpha_hat) == config["diffusion"]["num_steps"]
                assert len(csdi.alpha) == config["diffusion"]["num_steps"]


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])