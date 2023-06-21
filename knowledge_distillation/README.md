# Knowledge Distillation Loss Functions

Aknowledgement to [AberHu](https://github.com/AberHu) for his [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) repository.

Pytorch implementation of various Knowledge Distillation methods. 


## Lists
  Name | Method | Paper Link | Code Link
  :---- | ----- | :----: | :----:
  Logits   | mimic learning via regressing logits | [paper](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf) | [code](logits.py)
  ST       | soft target | [paper](https://arxiv.org/pdf/1503.02531.pdf) | [code](st.py)
  AT       | attention transfer | [paper](https://arxiv.org/pdf/1612.03928.pdf) | [code](at.py)
  Fitnet   | hints for thin deep nets | [paper](https://arxiv.org/pdf/1412.6550.pdf) | [code](fitnet.py)
  NST      | neural selective transfer | [paper](https://arxiv.org/pdf/1707.01219.pdf) | [code](nst.py)
  PKT      | probabilistic knowledge transfer | [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | [code](pkt.py)
  FSP      | flow of solution procedure | [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) | [code](fsp.py)
  FT       | factor transfer | [paper](http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf) | [code](ft.py)
  RKD      | relational knowledge distillation | [paper](https://arxiv.org/pdf/1904.05068.pdf) | [code](rkd.py)
  AB       | activation boundary | [paper](https://arxiv.org/pdf/1811.03233.pdf) | [code](ab.py)
  SP       | similarity preservation | [paper](https://arxiv.org/pdf/1907.09682.pdf) | [code](sp.py)
  Sobolev  | sobolev/jacobian matching | [paper](https://arxiv.org/pdf/1706.04859.pdf) | [code](sobolev.py)
  BSS      | boundary supporting samples | [paper](https://arxiv.org/pdf/1805.05532.pdf) | [code](bss.py)
  CC       | correlation congruence | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf) | [code](cc.py)
  LwM      | learning without memorizing | [paper](https://arxiv.org/pdf/1811.08051.pdf) | [code](lwm.py)
  IRG      | instance relationship graph | [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf) | [code](irg.py)
  VID      | variational information distillation | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) | [code](vid.py)
  OFD      | overhaul of feature distillation | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf) | [code](ofd.py)
  AFD      | attention feature distillation | [paper](https://openreview.net/pdf?id=ryxyCeHtPB) | [code](afd.py)
  CRD      | contrastive representation distillation | [paper](https://openreview.net/pdf?id=SkgpBJrtvS) | [code](crd.py)
  DML      | deep mutual learning | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) | [code](dml.py)