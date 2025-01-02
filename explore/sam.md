# Segment Anything

![image](https://github.com/user-attachments/assets/c8924a90-157f-459c-873f-570a99a1ec32)

1.Train

    #1. Image encoder
    x = Conv2d(kernel_size=16, stride=16)(img)  #embedding to patches [B,H/16,W/16,C]
    pos_embed = Parameter(zeros(1, H // 16, W // 16, C)  #init absolute positional embedding
    #image_embed: final image embedding [B,embed,H/16,W/16]
    #high_res_feats: inter layers embeddings [[B,embed/8,H/4,W/4],[B,embed/4,H/8,W/8]]
    [image_embed,high_res_feats] = ViT(x + pos_embed)  

    # image embed position encoding
    pos_embed_gaussian_matrix = randn(2, embed_dim/2)
    image_pe = #[1,embed,H/16,W/16]
    grid = ones((H/16,W/16)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5 #[H/16,W/16]
    y_embed = y_embed / (H/16)
    x_embed = x_embed / (W/16)
    coords = stack([x_embed, y_embed], dim=-1) #[H/16,W/16,2]  
    coords = 2 * coords - 1
    coords = coords @ positional_encoding_gaussian_matrix #[H/16,W/16,embed_dim/2]  
    image_embed_pe = cat([sin(2*pi*coords),cos(2*pi*coords)],dim=-1) #[1,embed,H/16,W/16]

    #2. Prompt encoder
    point_embeddings = [Embedding(1, embed_dim),Embedding(1, embed_dim),
                        Embedding(1, embed_dim),Embedding(1, embed_dim)]
    pos_embed_gaussian_matrix = randn(2, embed_dim/2)
    ## 2.0 sample points (sample from gt masks for example)
    see next part details
    

    ## 2.1 encode points 
    # points[B,num_pts,2], format like [[[x, y],..,[x,y],[0,0]]] unnormalized within [1024,1024]
    # labels[B,num_pts], format like [[1,-1]],indicating which points are valid (some points may padded)
    points = (points + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    points = points * 2 - 1 # Shift to [-1,1]
    points_embed = points @ pos_embed_gaussian_matrix #[B,num_pts,embed_dim/2]
    points_embed = cat([sin(2*pi*points_embed), cos(2*pi*points_embed)],
                       dim=-1) #[B,num_pts,embed_dim]
    points_embed[labels==-1] = 0.0 #reset padding point to zero #[B,num_pts,embed_dim]

    
    ## 2.2 encode box
    # box[B,2,2], format like [[[x, y],[x',y']]] unnormalized within [1024,1024]
    box = (box + 0.5) / 1024  # Shift to center of pixel and normalize to [0,1]
    box = box * 2 - 1 # Shift to [-1,1]
    box_embed = box @ pos_embed_gaussian_matrix #[B,2,embed_dim/2]
    box_embed = cat([sin(2*pi*box_embed), cos(2*pi*box_embed)],
                       dim=-1) #[B,2,embed_dim]
    box_embed[:,0,:] += point_embeddings[2]
    box_embed[:,1,:] += point_embeddings[3] #[B,2,embed_dim]

    ## 2.3 encode mask
    dense_prompt_embed =     #[B,embed_dim,H/16,W/16]

    #[B,n,embed_dim], where where n depends on points/box/mask avaliablity
    sparse_prompt_embed = cat([points_embed, box_embed], dim=1) 

    # 3. Mask decoder
    #sparse_prompt_embed: [B,n,embed_dim], n=0/2/4
    #dense_prompt_embed: [B,embed_dim,H/16,W/16]
    #image_embed: [B,embed_dim,H/16,W/16]
    #image_embed_pe: [1,embed,H/16,W/16]
    #cls is the number of labels
    #high_res_feats: [[B,embed/4,H/8,W/8],[[B,embed/8,H/4,W/4]]
    
    num_multimask_outputs = 3  #each point predict 3 to resolve ambigurity
    num_mask_tokens = num_multimask_outputs + 1
    iou_token = Embedding(1, embed_dim)
    mask_tokens = Embedding(num_mask_tokens, embed_dim)
    obj_score_token = Embedding(1, embed_dim)
    output_tokens = cat([obj_score_token,iou_token,mask_tokens], 
                        dim=0).unsequeeze().expand() #[B,6,embed_dim]         
    prompt_embed = cat((output_tokens, sparse_prompt_embed), dim=1) #[B,6+n,embed_dim]  
    
    # Expand per-image to be per-mask, [B,embed_dim,H/16,W/16]
    # src = repeat_interleave(image_embed,cls,dim=0)  #[B*cls,embed_dim,H/16,W/16]
    
    image_embed = image_embed + dense_prompt_embed  #[B,embed_dim,H/16,W/16]
    image_embed_pe =repeat_interleave(image_embed_pe,B,dim=0)  #[B,embed_dim,H/16,W/16]

    # promptF[B,6+n,embed_dim], imageF[B,H/16*W/16,embed_dim]
    promptF,imageF=TwoWayTransformer(image_embedd, image_embed_pe, prompt_embed):
        image_embed = image_embed.flatten(2).permute(0, 2, 1) #[B,H/16*W/16,embed_dim]
        image_embed_pe=image_embed_pe.flatten(2).permute(0,2,1)#[B,H/16*W/16,embed_dim]
        query = prompt_embed
        key = image_embed
        query_pe=prompt_embed
        key_pe = image_embed_pe

        # cross attention between image embedding and prompt embedding
        for i in 4: 
            query, key= TwoWayAttentionBlock():
                # Self attention on q
                q = query + query_pe
                attn_out = self_attn(q=q, k=q, v=query)
                query = query + attn_out
    
                # Cross attention, prompt emedding attending to image embedding
                q = query + query_pe
                k = key + key_pe
                attn_out = cross_attn(q=q, k=k, v=key)
                query = query + attn_out
                query = mlp(query) + query
    
                # Cross attention, image embedding attending to prompt emedding
                q = query + query_pe
                k = key + key_pe
                attn_out = cross_attn(q=k, k=q, v=query)
                key = key + attn_out

        # Apply the final attention layer from the prompt to image
        q = query + prompt_embed
        k = key + image_embed_pe
        attn_out = cross_attn(q=q, k=k, v=key)
        query = query + attn_out

    iou_token_out = promptF[:,1,:] #[B,1,embed_dim]
    mask_tokens_out = promptF[:,2:2+num_mask_tokens,:] #[B,4,embed_dim]
    # Upscale mask embeddings and predict masks using the mask tokens
    imageF = imageF.rearrage() #[B,embed_dim,H/16,W/16]
    upscaled_embed = ConvTranspose2D(embed_dim,embed_dim/4,
                        kernel=2,stride=2)(imageF) 
    upscaled_embed = upscaled_embed + high_res_feats[1] #[B,embed_dim/4,H/8,W/8]
    upscaled_embed = ConvTranspose2D(embed_dim/4,embed_dim/8,
                        kernel=2,stride=2)(upscaled_embed) 
    upscaled_embed = upscaled_embed + high_res_feats[0] #[B,embed_dim/8,H/4,W/4]

    #output
    out = [mlps[i](mask_tokens_out[:,i,:]) for i in range(num_mask_tokens))]
    out = stack(out, dim=1) #[B,4,embed_dim/8]
    masks = (out @ upscaled_embed.view(B,embed_dim/8,H/4*W/4)) #[B,4,H/4*W/4]
    masks = masks.view(..)[:,1:,:,:] #[B,3,H/4,W/4], takes only 3 as multi-masks
    masks = F.interpolate(masks, (H,W)) #[B,3,H,W]



2.Points sampling

    #gt_masks[B*3,1,H,W] each image has 3 masks, num_pt=1
    pred_masks = zeros(B*3,1,H,W)
    
    #false positive region, a new point sampled in this region should have
    #negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks #[B*3,1,H,W]

    #false negative region, a new point sampled in this region should have
    #positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks #[B*3,1,H,W]

    #whether the prediction completely match the ground-truth on each mask
    all_correct = (gt_masks == pred_masks).flatten(2) #[B*3, 1, H*W]
    all_correct = torch.all(all_correct, dim=2) #[B*3,1]
    all_correct = all_correct[..., None, None] #[B*3,1,1,1]

    #channel 0 is FP map, while channel 1 is FN map. In case the predictions are all 
    #correct (no FP or FN), sample a negative point from the background region
    pts_noise = rand(B*3, num_pt, H, W, 2)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2) #(B*3,num_pt)
    labels = (pts_idx % 2) #(B*3, num_pt)
    pts_idx = pts_idx // 2 #(B*3, num_pt)
    pts_x = pts_idx % W
    pts_y = pts_idx // W
    points = stack([pts_x, pts_y], dim=2)  #(B*3, num_pt, 2)

 3. Data
  
   
    
    

