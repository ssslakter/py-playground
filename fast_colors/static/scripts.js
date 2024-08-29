
document.addEventListener('DOMContentLoaded', () => {
  var imgInp = document.getElementById('file-inp');
  var imgPrev = document.getElementById('preview');

  imgInp.onchange = () => {
    const [file] = imgInp.files;
    if (file) {
      imgPrev.src = URL.createObjectURL(file);
    }
  };

  var imgCards = document.querySelectorAll('.img-card');
  imgCards.forEach(card => {
    card.addEventListener('click', () => {
      imgPrev.src = card.querySelector('img').src;
    });
  });
  
});