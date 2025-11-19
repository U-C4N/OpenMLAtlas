const translations = {
    en: {
        title: "Basic Math",
        nav: { learn: "Learn", play: "Play", test: "Test" },
        sections: { learn: "Learn Math", play: "Play Game", test: "Quiz" },
        notes: [
            { title: "Addition (+)", content: "Addition is bringing two or more numbers together to make a new total." },
            { title: "Subtraction (-)", content: "Subtraction is taking one number away from another." },
            { title: "Multiplication (×)", content: "Multiplication is adding a number to itself a certain number of times." },
            { title: "Division (÷)", content: "Division is splitting into equal parts or groups." }
        ],
        game: { check: "Check", score: "Score", correct: "Correct!", wrong: "Try Again!", placeholder: "?" },
        quiz: { start: "Start Quiz", result: "You got {score} out of {total}!" }
    },
    es: {
        title: "Matemáticas Básicas",
        nav: { learn: "Aprender", play: "Jugar", test: "Prueba" },
        sections: { learn: "Aprender Matemáticas", play: "Jugar Juego", test: "Cuestionario" },
        notes: [
            { title: "Suma (+)", content: "La suma es juntar dos o más números para obtener un total nuevo." },
            { title: "Resta (-)", content: "La resta es quitar un número de otro." },
            { title: "Multiplicación (×)", content: "La multiplicación es sumar un número a sí mismo varias veces." },
            { title: "División (÷)", content: "La división es repartir en partes o grupos iguales." }
        ],
        game: { check: "Comprobar", score: "Puntaje", correct: "¡Correcto!", wrong: "¡Inténtalo de nuevo!", placeholder: "?" },
        quiz: { start: "Empezar Prueba", result: "¡Obtuviste {score} de {total}!" }
    },
    fr: {
        title: "Mathématiques de Base",
        nav: { learn: "Apprendre", play: "Jouer", test: "Test" },
        sections: { learn: "Apprendre les Maths", play: "Jouer au Jeu", test: "Quiz" },
        notes: [
            { title: "Addition (+)", content: "L'addition consiste à réunir deux nombres ou plus pour obtenir un nouveau total." },
            { title: "Soustraction (-)", content: "La soustraction consiste à enlever un nombre d'un autre." },
            { title: "Multiplication (×)", content: "La multiplication consiste à ajouter un nombre à lui-même un certain nombre de fois." },
            { title: "Division (÷)", content: "La division consiste à partager en parts ou groupes égaux." }
        ],
        game: { check: "Vérifier", score: "Score", correct: "Correct !", wrong: "Essaie encore !", placeholder: "?" },
        quiz: { start: "Commencer le Quiz", result: "Tu as eu {score} sur {total} !" }
    },
    ru: {
        title: "Основы Математики",
        nav: { learn: "Учить", play: "Играть", test: "Тест" },
        sections: { learn: "Учить Математику", play: "Играть в Игру", test: "Викторина" },
        notes: [
            { title: "Сложение (+)", content: "Сложение — это объединение двух или более чисел для получения новой суммы." },
            { title: "Вычитание (-)", content: "Вычитание — это отнятие одного числа от другого." },
            { title: "Умножение (×)", content: "Умножение — это сложение числа с самим собой определенное количество раз." },
            { title: "Деление (÷)", content: "Деление — это разделение на равные части или группы." }
        ],
        game: { check: "Проверить", score: "Счет", correct: "Правильно!", wrong: "Попробуй еще раз!", placeholder: "?" },
        quiz: { start: "Начать Тест", result: "Вы набрали {score} из {total}!" }
    },
    tr: {
        title: "Temel Matematik",
        nav: { learn: "Öğren", play: "Oyna", test: "Test" },
        sections: { learn: "Matematik Öğren", play: "Oyun Oyna", test: "Sınav" },
        notes: [
            { title: "Toplama (+)", content: "Toplama, iki veya daha fazla sayıyı bir araya getirerek yeni bir toplam oluşturmaktır." },
            { title: "Çıkarma (-)", content: "Çıkarma, bir sayıyı diğerinden eksiltmektir." },
            { title: "Çarpma (×)", content: "Çarpma, bir sayıyı kendisiyle belirli bir sayıda toplamaktır." },
            { title: "Bölme (÷)", content: "Bölme, eşit parçalara veya gruplara ayırmaktır." }
        ],
        game: { check: "Kontrol Et", score: "Puan", correct: "Doğru!", wrong: "Tekrar Dene!", placeholder: "?" },
        quiz: { start: "Sınavı Başlat", result: "{total} üzerinden {score} aldın!" }
    }
};

let currentLang = 'en';
let gameScore = 0;
let currentGameAnswer = 0;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    setLanguage('tr'); // Default to Turkish as per user request context (greeting was Turkish) or maybe EN? Let's default to TR since the user spoke TR.
    renderNotes();
    generateGameQuestion();
});

function setLanguage(lang) {
    currentLang = lang;
    
    // Update active button
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.lang === lang) btn.classList.add('active');
    });

    // Update text content
    const t = translations[lang];
    document.getElementById('app-title').textContent = t.title;
    document.getElementById('nav-learn').textContent = t.nav.learn;
    document.getElementById('nav-play').textContent = t.nav.play;
    document.getElementById('nav-test').textContent = t.nav.test;
    
    document.getElementById('learn-title').textContent = t.sections.learn;
    document.getElementById('play-title').textContent = t.sections.play;
    document.getElementById('test-title').textContent = t.sections.test;
    
    document.getElementById('game-submit').textContent = t.game.check;
    document.getElementById('game-score').textContent = `${t.game.score}: ${gameScore}`;
    document.getElementById('game-input').placeholder = t.game.placeholder;
    
    document.getElementById('start-quiz-btn').textContent = t.quiz.start;

    renderNotes();
    // If quiz is showing result, update it? Maybe too complex. Reset quiz text if not active.
}

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(sec => sec.classList.add('hidden'));
    document.getElementById(`${sectionId}-section`).classList.remove('hidden');
    
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`nav-${sectionId}`).classList.add('active');
}

function renderNotes() {
    const container = document.getElementById('math-notes');
    container.innerHTML = '';
    translations[currentLang].notes.forEach(note => {
        const div = document.createElement('div');
        div.className = 'note-card';
        div.innerHTML = `<h3>${note.title}</h3><p>${note.content}</p>`;
        container.appendChild(div);
    });
}

// Game Logic
function generateGameQuestion() {
    const ops = ['+', '-', '×']; // Simple ops for primary school
    const op = ops[Math.floor(Math.random() * ops.length)];
    let a = Math.floor(Math.random() * 10) + 1;
    let b = Math.floor(Math.random() * 10) + 1;
    
    if (op === '-') {
        if (a < b) [a, b] = [b, a]; // Ensure positive result
    }
    
    let question = `${a} ${op} ${b}`;
    if (op === '×') {
        currentGameAnswer = a * b;
    } else if (op === '+') {
        currentGameAnswer = a + b;
    } else {
        currentGameAnswer = a - b;
    }
    
    document.getElementById('game-question').textContent = `${question} = ?`;
    document.getElementById('game-input').value = '';
    document.getElementById('game-feedback').textContent = '';
    document.getElementById('game-input').focus();
}

function checkGameAnswer() {
    const input = document.getElementById('game-input');
    const val = parseInt(input.value);
    const t = translations[currentLang].game;
    
    if (val === currentGameAnswer) {
        document.getElementById('game-feedback').textContent = t.correct;
        document.getElementById('game-feedback').style.color = 'green'; // Or keep black for theme? Let's stick to text feedback.
        gameScore += 10;
        document.getElementById('game-score').textContent = `${t.score}: ${gameScore}`;
        setTimeout(generateGameQuestion, 1500);
    } else {
        document.getElementById('game-feedback').textContent = t.wrong;
        document.getElementById('game-feedback').style.color = 'red'; // Maybe just bold black?
    }
}

// Quiz Logic
const quizQuestions = [
    { q: "5 + 3 = ?", options: [8, 7, 9, 6], a: 8 },
    { q: "10 - 4 = ?", options: [5, 6, 4, 7], a: 6 },
    { q: "3 × 3 = ?", options: [6, 9, 12, 8], a: 9 },
    { q: "12 ÷ 2 = ?", options: [5, 6, 4, 8], a: 6 },
    { q: "7 + 6 = ?", options: [12, 14, 13, 11], a: 13 }
];

let currentQuizIndex = 0;
let quizScore = 0;

function startQuiz() {
    currentQuizIndex = 0;
    quizScore = 0;
    document.getElementById('start-quiz-btn').classList.add('hidden');
    document.getElementById('quiz-question-area').classList.remove('hidden');
    document.getElementById('quiz-result').textContent = '';
    showQuizQuestion();
}

function showQuizQuestion() {
    if (currentQuizIndex >= quizQuestions.length) {
        finishQuiz();
        return;
    }
    
    const q = quizQuestions[currentQuizIndex];
    document.getElementById('quiz-question').textContent = q.q;
    const optsDiv = document.getElementById('quiz-options');
    optsDiv.innerHTML = '';
    
    q.options.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = opt;
        btn.onclick = () => checkQuizAnswer(opt, q.a);
        optsDiv.appendChild(btn);
    });
}

function checkQuizAnswer(selected, correct) {
    if (selected === correct) {
        quizScore++;
    }
    currentQuizIndex++;
    showQuizQuestion();
}

function finishQuiz() {
    document.getElementById('quiz-question-area').classList.add('hidden');
    document.getElementById('start-quiz-btn').classList.remove('hidden');
    document.getElementById('start-quiz-btn').textContent = translations[currentLang].game.wrong.replace("!", "") + " / " + translations[currentLang].nav.test; // Restart text
    
    const t = translations[currentLang].quiz;
    const resultText = t.result.replace('{score}', quizScore).replace('{total}', quizQuestions.length);
    document.getElementById('quiz-result').textContent = resultText;
}
