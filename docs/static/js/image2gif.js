document.querySelectorAll('.gif-hover-image').forEach(img => {
    const originalSrc = img.src; // 保存原始静态图片路径
    const gifSrc = img.dataset.gif; // 获取 data-gif 的路径
  
    // 鼠标悬浮事件
    img.addEventListener('mouseover', () => {
      img.src = gifSrc; // 切换为对应 GIF
    });
  
    // 鼠标移开事件
    img.addEventListener('mouseout', () => {
      img.src = originalSrc; // 恢复为静态图片
    });
  });
  