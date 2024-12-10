const scrollContainer = document.querySelector('.scroll-container');
let isUserInteracting = false;
let autoScrollInterval;
let scrollSpeed = 0.8;

function autoScroll() {
  if (!isUserInteracting) {
    scrollContainer.scrollLeft += scrollSpeed; 
    if (scrollContainer.scrollLeft >= scrollContainer.scrollWidth - scrollContainer.offsetWidth) {
      scrollContainer.scrollLeft = 0;
    }
  }
}

scrollContainer.addEventListener('mousedown', () => {
  isUserInteracting = true;
});

scrollContainer.addEventListener('mouseup', () => {
  isUserInteracting = false;
});

function startAutoScroll() {
  autoScrollInterval = setInterval(autoScroll, 20); 
}


function stopAutoScroll() {
  clearInterval(autoScrollInterval);
}


scrollContainer.addEventListener('mouseover', stopAutoScroll);

scrollContainer.addEventListener('mouseout', startAutoScroll);


// 初始化
startAutoScroll();