import logging
from modules import *


class EntityEncoder(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_input=0.3, finetune=False,
                 dropout_neighbors=0.0,
                 device=torch.device("cpu"),BiLSTM_hidden_size = 100, Bilstm_num_layers = 2, 
                 Bilstm_seq_length = 3, BiLSTM_input_size = 100, max_rel = 10, max_tail = 10):
        super(EntityEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)
        self.num_symbols = num_symbols

        self.BiLSTM_hidden_size = BiLSTM_hidden_size
        self.Bilstm_num_layers = Bilstm_num_layers
        self.Bilstm_seq_length = Bilstm_seq_length
        self.BiLSTM_input_size = BiLSTM_input_size

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout_input)
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.NeighborAggregator = AttentionSelectContext(dim=embed_dim, dropout=dropout_neighbors, BiLSTM_hidden_size = BiLSTM_hidden_size, 
                                                         Bilstm_num_layers = Bilstm_num_layers, Bilstm_seq_length = Bilstm_seq_length, 
                                                         BiLSTM_input_size = BiLSTM_input_size, max_rel = max_rel, max_tail = max_tail)

    def neighbor_encoder_mean(self, connections, num_neighbors):
        """
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        """
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        return out.tanh()

    def neighbor_encoder_soft_select(self, connections_left, connections_right, head_left, head_right):
        """
        :param connections_left: [b, max, 2]
        :param connections_right:
        :param head_left:
        :param head_right:
        :return:
        """
        #connection_left (5,10,10,2)
        #head_left (5,100)
        
        relations_left = connections_left[:, :,:, 0].squeeze(-1)#(5,10,10)
        entities_left = connections_left[:, :,:, 1].squeeze(-1)#(5,10,10)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # [b, max, dim]#(5,10,10,100)
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))#(5,10,10,100)

        rel_embeds_left = torch.mean(rel_embeds_left, dim=2)#(5,10,100), the representation of neighbor relations
        ent_embeds_left = torch.mean(ent_embeds_left, dim=2)#(5,10,100), the representation of the one-hop neighbors from the perspective of the head entity.
        # rel_embeds_left = F.normalize(rel_embeds_left, dim=2)
        # ent_embeds_left = F.normalize(ent_embeds_left, dim=2)

        # for rel_left in range(relations_left.shape[0]):
        #     rel_left_set_ = [relations_left[rel_left][x].item() for x in range(relations_left[rel_left].shape[0])]
        #     rel_left_set = []
        #     [rel_left_set.append(x) for x in rel_left_set_ if x not in rel_left_set]
        #     if self.pad_idx in rel_left_set:
        #         rel_left_set.remove(self.pad_idx)
        #     rel_left_set = list(rel_left_set)
        #     for index in range(10):
        #         if(index<len(rel_left_set)):
        #             rel_type_l = rel_left_set[index]
        #             sample_l = torch.nonzero(relations_left[rel_left] == rel_type_l)
        #             id_low = sample_l[0, 0].item()
        #             id_high = sample_l[sample_l.shape[0]-1, 0].item() + 1
        #             ent_embeds_left[rel_left, index] = torch.mean(ent_embeds_left[rel_left,id_low:id_high], dim=0)
        #             rel_embeds_left[rel_left, index] = rel_embeds_left[rel_left, id_low]
        #             relations_left[rel_left, index] = rel_type_l
        #         else:
        #             ent_embeds_left[rel_left, index] = self.dropout(self.symbol_emb(self.pad_tensor))
        #             rel_embeds_left[rel_left, index] = self.dropout(self.symbol_emb(self.pad_tensor))
        #             relations_left[rel_left, index] = self.pad_tensor
        # rel_embeds_left = rel_embeds_left[:,:10,:]
        # ent_embeds_left = ent_embeds_left[:,:10,:]
        # relations_left = relations_left[:,:10]
        # entities_left = entities_left[:,:10]

        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        mask_matrix_left = torch.eq(relations_left, pad_matrix_left).squeeze(-1)  # [b, max] (5,10,10)

        relations_right = connections_right[:, :, 0].squeeze(-1)
        entities_right = connections_right[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (batch, 200, embed_dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right))  # (batch, 200, embed_dim)

        rel_embeds_right = torch.mean(rel_embeds_right, dim=2)
        ent_embeds_right = torch.mean(ent_embeds_right, dim=2)
        # rel_embeds_right = F.normalize(rel_embeds_right, dim=2)
        # ent_embeds_right = F.normalize(ent_embeds_right, dim=2)

        # for rel_right in range(relations_right.shape[0]):
        #     rel_right_set_ = [relations_right[rel_right][x].item() for x in range(relations_left[rel_right].shape[0])]
        #     rel_right_set = []
        #     [rel_right_set.append(x) for x in rel_right_set_ if x not in rel_right_set]
        #     if self.pad_idx in rel_right_set:
        #         rel_right_set.remove(self.pad_idx)
        #     rel_right_set = list(rel_right_set)
        #     for index in range(10):
        #         if(index<len(rel_right_set)):
        #             rel_type_r = rel_right_set[index]
        #             sample_r = torch.nonzero(relations_right[rel_right] == rel_type_r)
        #             id_low = sample_r[0, 0].item()
        #             id_high = sample_r[sample_r.shape[0]-1, 0].item() + 1
        #             ent_embeds_right[rel_right, index] = torch.mean(ent_embeds_right[rel_right,id_low:id_high], dim=0)
        #             rel_embeds_right[rel_right, index] = rel_embeds_right[rel_right, id_low]
        #             relations_right[rel_right, index] = rel_type_r
        #         else:
        #             ent_embeds_right[rel_right, index] = self.dropout(self.symbol_emb(self.pad_tensor))
        #             rel_embeds_right[rel_right, index] = self.dropout(self.symbol_emb(self.pad_tensor))
        #             relations_right[rel_right, index] = self.pad_tensor
        # rel_embeds_right = rel_embeds_right[:,:10,:]
        # ent_embeds_right = ent_embeds_right[:,:10,:]
        # relations_right = relations_right[:,:10]
        # entities_right = entities_right[:,:10]

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_matrix_right = torch.eq(relations_right, pad_matrix_right).squeeze(-1)  # [b, max]

        left = [head_left, rel_embeds_left, ent_embeds_left] #head_left为(5,100) (5,10,100) (5,10,100)
        right = [head_right, rel_embeds_right, ent_embeds_right]
        output = self.NeighborAggregator(left, right, mask_matrix_left, mask_matrix_right)

        return output

    def forward(self, entity, entity_meta=None):
        '''
         query: (batch_size, 2)
         entity: (few, 2)
         return: (batch_size, )
         '''
        if entity_meta is not None:
            entity = self.symbol_emb(entity) #(5,2,100)
            entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta
            #(5,10,10,2)  (5) 
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            #(5,1,100)
            entity_left = entity_left.squeeze(1)#(5,100)
            entity_right = entity_right.squeeze(1)#(5,100)
            en_h, entity_left, entity_right = self.neighbor_encoder_soft_select(entity_left_connections,
                                                                          entity_right_connections,
                                                                          entity_left, entity_right)
        else:
            # no_meta
            entity = self.symbol_emb(entity)
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
            en_h = None

        return en_h, entity_left, entity_right #en_h(5,600) 


class RelationRepresentation(nn.Module):
    def __init__(self, emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=3,
                                                  with_pos=True)

    def forward(self, left, right):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return: [batch, dim]
        """

        relation = self.RelationEncoder(left, right)
        return relation


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # a = torch.ones_like(mask)
        # b = torch.arange(batch_size * anchor_count).view(-1, 1).to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Matcher(nn.Module): #改了
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_layers=0.1, dropout_input=0.3,
                 dropout_neighbors=0.0,
                 finetune=False, num_transformer_layers=6, num_transformer_heads=4,
                 device=torch.device("cpu"), lam = 0.1, BiLSTM_hidden_size = 100, Bilstm_num_layers = 2, Bilstm_seq_length = 3, BiLSTM_input_size = 100,
                 max_rel = 10, max_tail = 10):
        super(Matcher, self).__init__()
        self.EntityEncoder = EntityEncoder(embed_dim, num_symbols,
                                           use_pretrain=use_pretrain,
                                           embed=embed, dropout_input=dropout_input,
                                           dropout_neighbors=dropout_neighbors,
                                           finetune=finetune, device=device, 
                                           BiLSTM_hidden_size = BiLSTM_hidden_size, 
                                           Bilstm_num_layers = Bilstm_num_layers, 
                                           Bilstm_seq_length = Bilstm_seq_length,
                                           BiLSTM_input_size = BiLSTM_input_size,
                                           max_rel = max_rel , max_tail = max_tail)
        self.RelationRepresentation = RelationRepresentation(emb_dim=embed_dim * 6,
                                                             num_transformer_layers=num_transformer_layers,
                                                             num_transformer_heads=num_transformer_heads,
                                                             dropout_rate=dropout_layers)
        self.device = device
        # self.batchsize = 128
        self.lam = lam
        self.BiLSTM_hidden_size = BiLSTM_hidden_size
        self.Bilstm_num_layers = Bilstm_num_layers
        self.Bilstm_seq_length = Bilstm_seq_length
        self.BiLSTM_input_size = BiLSTM_input_size
        self.criterion = nn.MarginRankingLoss(0.5)
        self.Prototype = SoftSelectPrototype(embed_dim * 6 * num_transformer_heads)

    def forward(self, support, query, false=None, isEval=False, support_meta=None, query_meta=None, false_meta=None):
        """
        :param support:
        :param query:
        :param false:
        :param isEval:
        :param support_meta:
        :param query_meta:
        :param false_meta:
        :return:
        """
        if not isEval:
            support_r = self.EntityEncoder(support, support_meta) #support_r[0]--[2] (5,600)
            query_r = self.EntityEncoder(query, query_meta)#support_r[0]--[2] (128,600)
            false_r = self.EntityEncoder(false, false_meta)#support_r[0]--[2] (128,600)

            # tail_label = torch.cat((support[:, 0], query[:, 0])).squeeze()
            # contrast_1 = torch.cat((support_r[0], query_r[0]), dim=0)
            # contrast_2 = torch.cat((support_r[1], query_r[1]), dim=0)
            # features1 = torch.cat((contrast_1.unsqueeze(1), contrast_2.unsqueeze(1)), dim=1)

            # supconloss1 = self.supconloss(features1, labels=tail_label, mask=None)
            supconloss1 = 0

            pos_h_q = query_r[0]
            pos_z0_q = query_r[1]
            pos_z1_q = query_r[2]

            pos_loss_q = self.lam * torch.norm(pos_z0_q - pos_z1_q, p=2, dim=1) + \
                       torch.norm(pos_h_q[:, 0:2 * self.BiLSTM_hidden_size] +
                                  pos_h_q[:, 2 * self.BiLSTM_hidden_size:2 * 2 * self.BiLSTM_hidden_size] -
                                  pos_h_q[:, 2 * 2 * self.BiLSTM_hidden_size:2 * 3 * self.BiLSTM_hidden_size], p=2,
                                  dim=1)#shape为(128)
            
            neg_h_q = false_r[0]
            neg_z0_q = false_r[1]
            neg_z1_q = false_r[2]

            neg_loss_q = self.lam * torch.norm(neg_z0_q - neg_z1_q, p=2, dim=1) + \
                       torch.norm(neg_h_q[:, 0:2 * self.BiLSTM_hidden_size] +
                                  neg_h_q[:, 2 * self.BiLSTM_hidden_size:2 * 2 * self.BiLSTM_hidden_size] -
                                  neg_h_q[:, 2 * 2 * self.BiLSTM_hidden_size:2 * 3 * self.BiLSTM_hidden_size], p=2,
                                  dim=1)
            
            y = -torch.ones(neg_h_q.shape[0]).to(self.device) #(128)
            supconloss1 = self.criterion(pos_loss_q, neg_loss_q, y)

            support_r = self.RelationRepresentation(support_r[1], support_r[2])
            query_r = self.RelationRepresentation(query_r[1], query_r[2])
            false_r = self.RelationRepresentation(false_r[1], false_r[2])

            # support_r = support_r[0] + support_r[1]
            # query_r = query_r[0] + query_r[1]
            # false_r = false_r[0] + false_r[1]

            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)
        else:

            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)

            support_r = self.RelationRepresentation(support_r[1], support_r[2])
            query_r = self.RelationRepresentation(query_r[1], query_r[2])

            # support_r = support_r[0] + support_r[1]
            # query_r = query_r[0] + query_r[1]

            center_q = self.Prototype(support_r, query_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = None
            supconloss1 = None
        return positive_score, negative_score, supconloss1
