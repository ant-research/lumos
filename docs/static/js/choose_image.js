
const thumbnails = document.querySelectorAll('.image-list img');
const selectedImage = document.querySelector('.selected-image img');
const NewViewImages = document.querySelectorAll('.new-view-container img');

const NewViewSets = [
    [
        'static/images/i23d/1-1.png', 'static/images/i23d/1-2.png', 'static/images/i23d/1-3.png',
        'static/images/i23d/1-4.png', 'static/images/i23d/1-5.png', 'static/images/i23d/1-6.png',
    ],
    [
        'static/images/i23d/2-1.png', 'static/images/i23d/2-2.png', 'static/images/i23d/2-3.png',
        'static/images/i23d/2-4.png', 'static/images/i23d/2-5.png', 'static/images/i23d/2-6.png',
    ],
    [
        'static/images/i23d/3-1.png', 'static/images/i23d/3-2.png', 'static/images/i23d/3-3.png',
        'static/images/i23d/3-4.png', 'static/images/i23d/3-5.png', 'static/images/i23d/3-6.png',
    ],
    [
        'static/images/i23d/4-1.png', 'static/images/i23d/4-2.png', 'static/images/i23d/4-3.png',
        'static/images/i23d/4-4.png', 'static/images/i23d/4-5.png', 'static/images/i23d/4-6.png',
    ]
];


thumbnails.forEach(thumbnail => {
    thumbnail.addEventListener('click', () => {
        const largeSrc = thumbnail.getAttribute('data-large');        
        
        selectedImage.src = largeSrc;

        thumbnails.forEach(img => img.classList.remove('selected'));

        thumbnail.classList.add('selected');

        if (largeSrc=="static/images/i23d/1.png"){
            NewViewImages.forEach((img, index)=>{
                img.src = NewViewSets[0][index]
            });
        } else if (largeSrc=="static/images/i23d/2.png"){
            NewViewImages.forEach((img, index)=>{
                img.src = NewViewSets[1][index]
            });
        } else if (largeSrc=="static/images/i23d/3.png"){
            const new_view_images = NewViewSets[2];
            NewViewImages.forEach((img, index)=>{
                img.src = new_view_images[index]
            });
        } else if (largeSrc=="static/images/i23d/4.png"){
            const new_view_images = NewViewSets[3];
            NewViewImages.forEach((img, index)=>{
                img.src = new_view_images[index]
            });
        }

    });
});
