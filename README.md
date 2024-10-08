# VitaGlyph: Vitalizing Artistic Typography with Flexible Dual-branch Diffusion Models
#### Kailai Feng, Yabo Zhang, Haodong Yu, Zhilong Ji, Jinfeng Bai, Hongzhi Zhang<sup>*</sup>, Wangmeng Zuo 
This repository is the official PyTorch implementation of "VitaGlyph: Vitalizing Artistic Typography with Flexible Dual-branch Diffusion Models".

## ✨ News/TODO
- [x] Source code of Controllable Compositional Generation.
- [ ] Source code of Semantic Typography
- [ ] Source code of Regional Decomposition
- [ ] More results


<!-- 
## 🖼️ Resluts

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Inbetweening results</td>
    </tr>
  <tr>
  <td>
    <img src=assets/input1_0.png width="250">
  </td>
  <td>
    <img src=assets/input1_1.png width="250">
  </td>
  <td>
    <img src=assets/ours1.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/input2_0.png width="250">
  </td>
  <td>
    <img src=assets/input2_1.png width="250">
  </td>
  <td>
    <img src=assets/ours2.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/input3_0.png width="250">
  </td>
  <td>
    <img src=assets/input3_1.png width="250">
  </td>
  <td>
    <img src=assets/ours3.gif width="250">
  </td>
  </tr> 
</table>
 -->

## 📖 Overview



<p align="center">
  <img src="assets/model.png" alt="model architecture" width="800"/>
  </br>
  An overview of the pipeline.
</p>




## ⚙️ Run inference demo
1) Run the following command to get inbetweening results.
``` shell
python ConComGen.py --resulation 1024 
```

<!-- You can change 'xN' to get arbitrary frame rates results. The reuslts are saved in the folder './output'. -->