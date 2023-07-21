from this import d
import torch
import torch.nn as nn
from functools import partial
import math
import torch.nn.functional as F
from nonlocal_cross_attention import NonCrossAttention

from timm.models.vision_transformer import Mlp
from timm.models.layers import trunc_normal_, DropPath

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.business_layer = []
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.business_layer.append(self.fc_q)
        self.business_layer.append(self.fc_k)
        self.business_layer.append(self.fc_v)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
            self.business_layer.append(self.ln0)
            self.business_layer.append(self.ln1)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.business_layer.append(self.fc_o)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_view=3, k=75, ln=False):
        super(ISAB, self).__init__()
        self.num_view = num_view
        self.k=k

        self.business_layer = []
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.business_layer += self.mab0.business_layer
        self.business_layer += self.mab1.business_layer

    def forward(self, X):
        cls_cat = X[:,-self.num_view*self.k:,...]
        X = X[:,:-self.num_view*self.k,...]

        H = self.mab0(cls_cat, X)
        return self.mab1(X, H)


class ISAB_seperate(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(ISAB_seperate, self).__init__()
        self.business_layer = []
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.business_layer += self.mab0.business_layer
        self.business_layer += self.mab1.business_layer

    def forward(self, X,cls1,cls2):
        '''
        Input:X(B,C,D,H,W), cls1(B,C), cls2(B,C)
        '''
        B,C,D,H,W = X.shape
        X = X.flatten(2).transpose(1,2)
        cls_cat = torch.cat((cls1.unsqueeze(1),cls2.unsqueeze(1)),dim=1) # (B,2,C)
        h = self.mab0(cls_cat, X)
        O = self.mab1(X, h)
        return O.transpose(1,2).reshape(B,C,D,H,W)
 
class ISAB_origin(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, attention_modes=None):
        super(ISAB_origin, self).__init__()
        self.business_layer = []
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        
        self.business_layer.append(self.I)
        nn.init.xavier_uniform_(self.I)   
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.business_layer.append(self.mab0)
        self.business_layer.append(self.mab1)

    def forward(self, X):
        '''
        Input:  X(B,N,C)
        Output: X (B,N,C)
        self attn
        '''
         #(B,N,C)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        X = self.mab1(X, H)
        return X

class Class_Cross_Attention_V1(nn.Module):
    # with slight modifications to do cross attn
    def __init__(self, dim, num_heads=8,num_view=3, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,\
                update_cls="old", voxel_size = 196, k=75):
        super().__init__()
        self.business_layer = []
        self.update_cls = update_cls
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_view = num_view
        self.voxel_size = voxel_size
        self.k=k

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_pro = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.business_layer.append(self.q)
        self.business_layer.append(self.k_pro)
        self.business_layer.append(self.v)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.business_layer.append(self.proj)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mhsa = MAB(dim, dim, dim, num_heads)
        self.business_layer += self.mhsa.business_layer

        self.conv_ffn = nn.Sequential(           
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), 
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), 
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, bias=False), 
         )
        self.business_layer.append(self.conv_ffn)

        self.avgpool = nn.AvgPool2d(kernel_size=(self.voxel_size, 1))
        self.business_layer.append(self.avgpool)
    
    def forward(self, x):
        '''
        x includes :semantic2_wo_cls, cls1, cls2, cls3=(B,N+2,C)
        semantic2_wo_cls: (B,N,C)
        cls: (B,1,C)
        '''
        cls_cat = x[:,-self.num_view*self.k:,...]
        semantic2_wo_cls = x[:,:-self.num_view*self.k,...]

        B, N, C = semantic2_wo_cls.shape
        q = self.q(cls_cat).reshape(B, self.num_view*self.k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        k = self.k_pro(semantic2_wo_cls).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(semantic2_wo_cls).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 
        hidden_state = attn.transpose(-2,-1)[:,:,:,:,None] @ v[:,:,:,None,:] #(B,N,2,C) 
        hidden_state = hidden_state.permute(0,2,3,1,4).flatten(3)  

        if self.update_cls == "new":
            # --1.update hidden state
            hidden_state = hidden_state.permute(0,3,1,2) #(B,C,N,2)
            hidden_state = self.conv_ffn(hidden_state) #(B,C,N,2)
            hidden_state = hidden_state.flatten(2).transpose(1,2) # (B,N*2,C)

            # --2. update cls token ,maxpooling
            cls_new = self.avgpool(hidden_state.view(B, N, self.num_view*self.k,C).permute(0,3,1,2)) #(B,C,N,2)-> (B,C,1,2)
            cls_cat = cls_cat + cls_new.flatten(2).transpose(1,2)

            # --3. cross. q:patch, kv:cls
            out = self.mhsa(semantic2_wo_cls, cls_cat) # (B,N,C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out
        elif self.update_cls == "old":
            out = self.mhsa(semantic2_wo_cls, cls_cat) # (B,N,C)
            return out

        semantic2_wo_cls = semantic2_wo_cls[:,:,None,:] 

        out = self.mhsa(semantic2_wo_cls.reshape(B,1*N,C),hidden_state.reshape(B,self.num_view*N,C),hidden_state.reshape(B,self.num_view*N,C))[0]

        return out   

class Class_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=75):
        super().__init__()
        self.business_layer = []
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k=k

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_pro = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.business_layer.append(self.q)
        self.business_layer.append(self.k_pro)
        self.business_layer.append(self.v)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.business_layer.append(self.proj)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:, 0:self.k ,...]).reshape(B, self.k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_pro(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.k, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls     
        
class LayerScale_Block_CA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4, k=75):
        super().__init__()
        self.business_layer = []
        self.norm1 = norm_layer(dim)
        self.business_layer.append(self.norm1)
        self.k=k
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, k=self.k)
        self.business_layer += self.attn.business_layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.business_layer.append(self.norm2)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.business_layer.append(self.mlp)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.business_layer.append(self.gamma_1)
        self.business_layer.append(self.gamma_2)

    
    def forward(self, x_cls, x_patch):
        u = torch.cat((x_cls, x_patch),dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls 
        
class Attention_talking_head(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
    
        attn = (q @ k.transpose(-2, -1))
        attn = self.proj_l(attn.permute(0,2,3,1)).permute(0,3,1,2)
                
        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Module):
    def __init__(self,  dim_in, dim_out, num_heads,num_inds, ln=False, attention_modes=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 Attention_block = ISAB_origin,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.business_layer = []
        dim=dim_in
        self.norm1 = norm_layer(dim)
        self.business_layer.append(self.norm1)

        self.attn = Attention_block(dim_in=256, dim_out=256, num_heads=num_heads, num_inds=num_inds, attention_modes='3')
        self.business_layer += self.attn.business_layer

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.business_layer.append(self.norm2)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.business_layer.append(self.mlp)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.business_layer.append(self.gamma_1)
        self.business_layer.append(self.gamma_2)
        

    def forward(self, x):        
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    

class FC_block_layers(nn.Module):
    def __init__(self, voxel_size=15*9*15, k=75):
        super().__init__()
        self.business_layer = []
        self.conv_layer  = nn.Conv3d(in_channels=256,out_channels=256,kernel_size=(13, 9, 13),padding=1)
        self.business_layer.append(self.conv_layer)
    def forward(self, x_cls, x_patch):
        '''
        x_cls:(B,k,C)
        x_patch:(B,N,C)
        '''
        B, _, C = x_patch.shape
        x_patch = x_patch.transpose(1,2).reshape(B, C, 15,9,15)
        y = self.conv_layer(x_patch) # (B, C, 5, 3, 5)
        x_cls = x_cls + y.flatten(2).transpose(1,2)

        return x_cls

class cait_models(nn.Module):
    def __init__(self, embed_dim=256, depth_SA=12, depth_CA=2, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 SA_block_layers = LayerScale_Block,
                 CA_block_layers = LayerScale_Block_CA,
                 act_layer=nn.GELU,
                 Attention_block = Attention_talking_head,Mlp_block=Mlp, init_scale=1e-4,
                Attention_block_token_only=Class_Attention, Mlp_block_token_only= Mlp, mlp_ratio_clstk = 4.0,
                num_inds=32,voxel_size=15*9*15, k=75, global_method = 'ca'):
        super().__init__()
        self.business_layer = []
        self.num_features = self.embed_dim = embed_dim  
        self.k=k
     
        self.ca_blocks = nn.ModuleList()
        # class attention to get the VIEW TOKEN
        if global_method == 'ca':
            for _ in range(depth_CA):
                tmp = CA_block_layers(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                    act_layer=act_layer, Attention_block=Attention_block_token_only,
                    Mlp_block=Mlp_block_token_only,init_values=init_scale, k=self.k)
                
                self.business_layer += tmp.business_layer
                self.ca_blocks.append(tmp)
        #  conv to get the VIEW TOKEN
        elif global_method == 'conv':
            tmp =  FC_block_layers(voxel_size, k)
            self.business_layer += tmp.business_layer
            self.ca_blocks.append(tmp)
            
        self.norm = norm_layer(embed_dim)
        self.business_layer.append(self.norm)
        self.apply(self._init_weights)

    def forward_features(self, x):
        '''
        input:  (cls,patch)=(B,1+N,C)
        output: (cls,patch)=(B,1+N,C)
        '''
        x_cls = x[:, 0:self.k ,...]  # (B,k,C)
        x_patch = x[:, self.k: ,...] # (B,N,C)
        
        for _ , blk in enumerate(self.ca_blocks): # 2 layers for CA
            x_cls = blk(x_cls, x_patch) 

        x = torch.cat((x_cls, x_patch), dim=1)
        # x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed','cls_token'}
        
  

class MultiViewBlock(nn.Module):
    def __init__(self, dim=256, num_branches=3, k=75):
        super(MultiViewBlock, self).__init__()
        self.business_layer = []
        self.num_branches = num_branches
        self.k = k
        self.sa_blocks=nn.ModuleList()
        self.cross_blocks=nn.ModuleList()
        for _ in range(num_branches):
            tmp1=[cait_models(embed_dim=dim, SA_block_layers=LayerScale_Block, depth_SA=5, depth_CA=2, num_heads=8, mlp_ratio=4, 
                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5, num_inds=16, voxel_size=15*9*15, k=self.k)]
            self.business_layer += tmp1[0].business_layer
            self.sa_blocks.append(nn.Sequential(*tmp1))

            tmp2=[ISAB(dim_in=dim, dim_out=dim, num_heads=8, num_view=num_branches-1, k=self.k)]
    
            self.business_layer += tmp2[0].business_layer
            self.cross_blocks.append(nn.Sequential(*tmp2))

    def forward(self, feature_list):
        '''
        input:[(cls,patch),  (cls,patch),  (cls,patch)]
        [(B,1+N,C),  (B,1+N,C),  (B,1+N,C)]
        '''
        #1.self attn
        feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.sa_blocks)]
        
        #2.cross attn
        cross_list=[]
        if self.num_branches==4:
            for i in range(self.num_branches):
                next1 = (i+1)% self.num_branches
                next2 = (i+2)% self.num_branches
                next3 = (i+3)% self.num_branches
                tmp = torch.cat((feature_list_sa[i][:,self.k:,...],feature_list_sa[next1][:,0:self.k,...],feature_list_sa[next2][:,0:self.k,...], feature_list_sa[next3][:,0:self.k,...] ), dim=1)
                tmp = self.cross_blocks[i](tmp)#return patch token (B,N,C)
                tmp = torch.cat((feature_list_sa[i][:,0:self.k,...],tmp), dim=1) #(B,k+N,C)
                cross_list.append(tmp) 
        elif self.num_branches==3:
            for i in range(self.num_branches):
                next1 = (i+1)% self.num_branches
                next2 = (i+2)% self.num_branches
                tmp = torch.cat((feature_list_sa[i][:,self.k:,...],feature_list_sa[next1][:,0:self.k,...],feature_list_sa[next2][:,0:self.k,...] ), dim=1)
                tmp = self.cross_blocks[i](tmp)#return patch token (B,N,C)
                tmp = torch.cat((feature_list_sa[i][:,0:self.k,...],tmp), dim=1) #(B,k+N,C)
                cross_list.append(tmp) 

        # skip conn
        cross_list = [feature_list[i]+cross_list[i] for i in range(self.num_branches)]
        return cross_list



class OneViewBlock(nn.Module):
    def __init__(self, dim=256, num_branches=4, k=75):
        super(OneViewBlock, self).__init__()
        self.business_layer = []
        self.num_branches = num_branches
        self.k = k
        self.conv_channel_proj_3d = nn.Conv3d(in_channels=256*4,out_channels=256,kernel_size=3,padding=1)

        self.nonlocal_cross_attention_0 = NonCrossAttention(256, inter_channels = 128,dimension = 3, sub_sample = False)
        self.nonlocal_cross_attention_1 = NonCrossAttention(256, inter_channels = 128,dimension = 3, sub_sample = False)
        self.nonlocal_cross_attention_2 = NonCrossAttention(256, inter_channels = 128,dimension = 3, sub_sample = False)
        self.nonlocal_cross_attention_3 = NonCrossAttention(256, inter_channels = 128,dimension = 3, sub_sample = False)
        self.business_layer += self.nonlocal_cross_attention_0.business_layer
        self.business_layer += self.nonlocal_cross_attention_1.business_layer
        self.business_layer += self.nonlocal_cross_attention_2.business_layer
        self.business_layer += self.nonlocal_cross_attention_3.business_layer
        if self.num_branches==4:
            self.conv_channel_proj = nn.Conv1d(in_channels=256*4,out_channels=256,kernel_size=3,padding=1)
            self.conv_channel_proj_1 = nn.Conv1d(in_channels=256*4,out_channels=256,kernel_size=3,padding=1)
        elif self.num_branches==3:
            self.conv_channel_proj = nn.Conv1d(in_channels=256*3,out_channels=256,kernel_size=3,padding=1)
        self.business_layer.append(self.conv_channel_proj)
        self.business_layer.append(self.conv_channel_proj_1)

        self.ca_blocks=nn.ModuleList()
        for _ in range(num_branches):
            tmp1=[cait_models(embed_dim=dim, SA_block_layers=LayerScale_Block, depth_SA=5, depth_CA=2, num_heads=8, mlp_ratio=4, 
                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5, num_inds=16, voxel_size=15*9*15, k=self.k, global_method = 'ca')]
        
            self.business_layer += tmp1[0].business_layer
            self.ca_blocks.append(nn.Sequential(*tmp1))
        
        self.conv_blocks=nn.ModuleList()
        for _ in range(num_branches):
          
            tmp1=[cait_models(embed_dim=dim, SA_block_layers=LayerScale_Block, depth_SA=5, depth_CA=2, num_heads=8, mlp_ratio=4, 
                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5, num_inds=16, voxel_size=15*9*15, k=self.k, global_method = 'conv')]
            self.business_layer += tmp1[0].business_layer
            self.conv_blocks.append(nn.Sequential(*tmp1))

    def forward(self, x, feature_list, key = 'patch', branch = 1):
        '''
        input:semantic2: (B,C,D,H,W)
                feature_list:
                    [(cls,patch),  (cls,patch),  (cls,patch)]
                    [(B,k+N,C),  (B,k+N,C),  (B,k+N,C)]
        '''
        B, C ,_,_,_= x[0].shape

        if key == 'patch' and branch == 4:
            x0 = self.nonlocal_cross_attention_0(x[0], feature_list) 
            x1 = self.nonlocal_cross_attention_1(x[1], feature_list) 
            x2 = self.nonlocal_cross_attention_2(x[2], feature_list)
            x3 = self.nonlocal_cross_attention_3(x[3], feature_list)
            y = torch.cat((x0, x1, x2, x3),dim=1 )
            y = self.conv_channel_proj_3d(y)
            y = y + x[0] 
            return y

        elif key == 'vt' and branch == 1:
            #1.self attn
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.sa_blocks)]
            #2.view token
            vts = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]
            if self.num_branches==4:
                vt_global = torch.cat((vts[0], vts[1], vts[2], vts[3]), dim=2).transpose(1,2) 
            elif self.num_branches==3:
                vt_global = torch.cat((vts[0], vts[1], vts[2]), dim=2).transpose(1,2) 
            vt_global = self.conv_channel_proj(vt_global) 
            vt_global = vt_global.reshape(B, C, 5, 3, 5) 
            #3.nonlocal            
            semantic2 = self.nonlocal_cross_attention(semantic2, vt_global)
            return semantic2

        elif key == 'vt_conv' and branch == 1:
            # 1. conv_global
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.conv_blocks)]
            conv_fs = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]

            # 2. vt_global
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.ca_blocks)]
            vts = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]

            #3. add
            outs = [vts[i] + conv_fs[i] for i in range(self.num_branches)]

            # cat
            if self.num_branches==4:
                vt_global = torch.cat((outs[0], outs[1], outs[2], outs[3]), dim=2).transpose(1,2) 
            elif self.num_branches==3:
                vt_global = torch.cat((outs[0], outs[1], outs[2]), dim=2).transpose(1,2) 
            vt_global = self.conv_channel_proj_1(vt_global) #(B,C,k)
            vt_global = vt_global.reshape(B, C, 5, 3, 5)

            #3.nonlocal
            semantic2 = self.nonlocal_cross_attention_0(x[0], vt_global) 
            return semantic2
            
        elif key == 'vt_conv_wo_cat' and branch == 1:
            # 1. conv_global
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.conv_blocks)]
            conv_fs = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]

            #2. vt_global
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.ca_blocks)]
            vts = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]

            #3. add
            outs = [vts[i] + conv_fs[i] for i in range(self.num_branches)] #(B,k,C)

            #(B,k,C)->(B, C, 5, 3, 5)
            outs = [outs[i].transpose(1,2).reshape(B, C, 5, 3, 5)  for i in range(self.num_branches)]

            #3.nonlocal
            x0 = self.nonlocal_cross_attention_0(x[0], outs[0]) 
            x1 = self.nonlocal_cross_attention_1(x[0], outs[1])
            x2 = self.nonlocal_cross_attention_2(x[0], outs[2]) 
            x3 = self.nonlocal_cross_attention_3(x[0], outs[3])
            y = torch.cat((x0, x1, x2, x3),dim=1 ) 
            y = self.conv_channel_proj_3d(y)
            y = y + x[0] 
            return y

        elif key == 'vt_conv' and branch == 4:
            # 1. conv_global
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.conv_blocks)]
            conv_fs = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]

            #2. vt_global
            feature_list_sa = [blk(x) for x, blk in zip(feature_list, self.ca_blocks)]
            vts = [feature_list_sa[i][:, 0:self.k ,...] for i in range(self.num_branches)]

            #3. add
            outs = [vts[i] + conv_fs[i] for i in range(self.num_branches)]

            if self.num_branches==4:
                vt_global = torch.cat((outs[0], outs[1], outs[2], outs[3]), dim=2).transpose(1,2) 
            elif self.num_branches==3:
                vt_global = torch.cat((outs[0], outs[1], outs[2]), dim=2).transpose(1,2) 
            vt_global = self.conv_channel_proj_1(vt_global)
            vt_global = vt_global.reshape(B, C, 5, 3, 5)

            #3.nonlocal
         
            x0 = self.nonlocal_cross_attention_0(x[0], vt_global)
            x1 = self.nonlocal_cross_attention_1(x[1], vt_global) 
            x2 = self.nonlocal_cross_attention_2(x[2], vt_global) 
            x3 = self.nonlocal_cross_attention_3(x[3], vt_global) 
            y = torch.cat((x0, x1, x2, x3),dim=1 )
            y = self.conv_channel_proj_3d(y)
            y = y + x[0] 
            return y
    

