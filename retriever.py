import torch
import torch.nn as nn
from loss import MultiLabelLoss


class DualEncoder(nn.Module):
    def __init__(self, mention_encoder,
                 entity_encoder,
                 type_loss):
        super(DualEncoder, self).__init__()
        self.mention_encoder = mention_encoder
        self.entity_encoder = entity_encoder
        self.loss_fct = MultiLabelLoss(type_loss)

    def encode(self, mention_token_ids=None,
               mention_masks=None,
               candidate_token_ids=None,
               candidate_masks=None,
               entity_token_ids=None,
               entity_masks=None):
        candidates_embeds = None
        mention_embeds = None
        entity_embeds = None
        # candidate_token_ids and mention_token_ids not None during training
        # mention_token_ids not None for embedding mentions during inference
        # entity_token_ids not None for embedding entities during inference
        if candidate_token_ids is not None:
            B, C, L = candidate_token_ids.size()
            candidate_token_ids = candidate_token_ids.view(-1, L)
            candidate_masks = candidate_masks.view(-1, L)
            # B X C X L --> BC X L
            candidates_embeds = self.entity_encoder(
                input_ids=candidate_token_ids,
                attention_mask=candidate_masks
            )[0][:, 0, :].view(B, C, -1)
        if mention_token_ids is not None:
            mention_embeds = self.mention_encoder(
                input_ids=mention_token_ids,
                attention_mask=mention_masks
            )[0][:, 0, :]
        if entity_token_ids is not None:
            # for getting all the entity embeddings
            entity_embeds = self.entity_encoder(input_ids=entity_token_ids,
                                                attention_mask=entity_masks)[
                                0][:, 0, :]
        return mention_embeds, candidates_embeds, entity_embeds

    def forward(self,
                mention_token_ids=None,
                mention_masks=None,
                candidate_token_ids=None,
                candidate_masks=None,
                passages_labels=None,
                entity_token_ids=None,
                entity_masks=None
                ):
        """

        :param inputs: [
                        mention_token_ids,mention_masks,  size: B X L
                        candidate_token_ids,candidate_masks, size: B X C X L
                        passages_labels, size: B X C
                        ]
        :return: loss, logits

        """
        if not self.training:
            return self.encode(mention_token_ids, mention_masks,
                               candidate_token_ids, candidate_masks,
                               entity_token_ids, entity_masks)
        B, C, L = candidate_token_ids.size()
        mention_embeds, candidates_embeds, _ = self.encode(
            mention_token_ids,
            mention_masks,
            candidate_token_ids,
            candidate_masks)
        mention_embeds = mention_embeds.unsqueeze(1)
        logits = torch.matmul(mention_embeds,
                              candidates_embeds.transpose(1, 2)).view(B, -1)
        loss = self.loss_fct(logits, passages_labels)

        return loss, logits
