// OpenMLAtlas Math Basics - Interactive 3D Learning App
// 2025 - THREE.js with WebGL Dithering

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// ============================================
// TRANSLATIONS - 6 Languages
// ============================================
const translations = {
    en: {
        langTitle: 'Select Language',
        menuTitle: 'Math Basics',
        menuSubtitle: 'Learn fundamental math operations in 3D',
        difficultyLabel: 'Difficulty:',
        easy: 'Easy',
        medium: 'Medium',
        hard: 'Hard',
        learn: 'Learn',
        learnDesc: 'Theory & 3D visualization',
        practice: 'Practice',
        practiceDesc: 'Interactive exercises',
        test: 'Test',
        testDesc: 'Quiz with scoring',
        selectOperation: 'Select Operation',
        addition: 'Addition',
        subtraction: 'Subtraction',
        multiplication: 'Multiplication',
        division: 'Division',
        learning: 'Learning',
        practicing: 'Practice',
        testing: 'Test',
        theory: 'Theory',
        nextExample: 'Next Example',
        check: 'Check',
        next: 'Next',
        score: 'Score:',
        time: 'Time:',
        question: 'Question:',
        testComplete: 'Test Complete!',
        correct: 'Correct',
        wrong: 'Wrong',
        tryAgain: 'Try Again',
        menu: 'Menu',
        excellent: 'Excellent! Perfect score!',
        great: 'Great job!',
        good: 'Good effort!',
        keepPracticing: 'Keep practicing!',
        theories: {
            add: 'Addition combines two or more numbers into a single sum. When you add, you are counting the total of all quantities together. Visualize it as merging groups of objects into one larger group.',
            sub: 'Subtraction finds the difference between two numbers. It tells you how much remains when you take away a quantity. Think of it as removing objects from a group.',
            mul: 'Multiplication is repeated addition. When you multiply A by B, you are adding A to itself B times. Visualize it as arranging objects in rows and columns to form a grid.',
            div: 'Division splits a number into equal parts. When you divide A by B, you are finding how many groups of B fit into A. Think of it as distributing objects equally.'
        }
    },
    tr: {
        langTitle: 'Dil Secin',
        menuTitle: 'Matematik Temelleri',
        menuSubtitle: '3D ile temel matematik islemlerini ogrenin',
        difficultyLabel: 'Zorluk:',
        easy: 'Kolay',
        medium: 'Orta',
        hard: 'Zor',
        learn: 'Ogren',
        learnDesc: 'Teori ve 3D gorsellestirme',
        practice: 'Pratik',
        practiceDesc: 'Interaktif alistirmalar',
        test: 'Test',
        testDesc: 'Puanli sinav',
        selectOperation: 'Islem Secin',
        addition: 'Toplama',
        subtraction: 'Cikarma',
        multiplication: 'Carpma',
        division: 'Bolme',
        learning: 'Ogrenme',
        practicing: 'Pratik',
        testing: 'Test',
        theory: 'Teori',
        nextExample: 'Sonraki Ornek',
        check: 'Kontrol Et',
        next: 'Sonraki',
        score: 'Puan:',
        time: 'Sure:',
        question: 'Soru:',
        testComplete: 'Test Tamamlandi!',
        correct: 'Dogru',
        wrong: 'Yanlis',
        tryAgain: 'Tekrar Dene',
        menu: 'Menu',
        excellent: 'Mukemmel! Tam puan!',
        great: 'Harika is!',
        good: 'Iyi calisma!',
        keepPracticing: 'Pratik yapmaya devam et!',
        theories: {
            add: 'Toplama, iki veya daha fazla sayiyi tek bir toplam halinde birlestirir. Toplama yaptiginizda, tum miktarlarin toplamini sayiyorsunuz. Bunu, nesne gruplarini daha buyuk bir grupta birlestirmek olarak dusunun.',
            sub: 'Cikarma, iki sayi arasindaki farki bulur. Bir miktari cikardiginizda ne kadar kaldigini soyler. Bunu bir gruptan nesne cikarmak olarak dusunun.',
            mul: 'Carpma, tekrarlanan toplamadir. A\'yi B ile carptiginizda, A\'yi B kez kendisine ekliyorsunuz. Bunu, nesneleri satirlar ve sutunlar halinde duzenleyerek bir izgara olusturmak olarak gorsellestirin.',
            div: 'Bolme, bir sayiyi esit parçalara ayirir. A\'yi B\'ye boldugunuzde, A\'nin icine kac B grubu sigdigini buluyorsunuz. Bunu nesneleri esit olarak dagitmak olarak dusunun.'
        }
    },
    ru: {
        langTitle: 'Выберите язык',
        menuTitle: 'Основы математики',
        menuSubtitle: 'Изучайте основные математические операции в 3D',
        difficultyLabel: 'Сложность:',
        easy: 'Легко',
        medium: 'Средне',
        hard: 'Сложно',
        learn: 'Учиться',
        learnDesc: 'Теория и 3D визуализация',
        practice: 'Практика',
        practiceDesc: 'Интерактивные упражнения',
        test: 'Тест',
        testDesc: 'Викторина с баллами',
        selectOperation: 'Выберите операцию',
        addition: 'Сложение',
        subtraction: 'Вычитание',
        multiplication: 'Умножение',
        division: 'Деление',
        learning: 'Обучение',
        practicing: 'Практика',
        testing: 'Тест',
        theory: 'Теория',
        nextExample: 'Следующий пример',
        check: 'Проверить',
        next: 'Далее',
        score: 'Счёт:',
        time: 'Время:',
        question: 'Вопрос:',
        testComplete: 'Тест завершён!',
        correct: 'Правильно',
        wrong: 'Неправильно',
        tryAgain: 'Попробовать снова',
        menu: 'Меню',
        excellent: 'Отлично! Идеальный результат!',
        great: 'Отличная работа!',
        good: 'Хорошая попытка!',
        keepPracticing: 'Продолжайте практиковаться!',
        theories: {
            add: 'Сложение объединяет два или более числа в одну сумму. Когда вы складываете, вы считаете общее количество всех величин вместе. Представьте это как слияние групп объектов в одну большую группу.',
            sub: 'Вычитание находит разницу между двумя числами. Оно показывает, сколько остаётся, когда вы убираете количество. Думайте об этом как об удалении объектов из группы.',
            mul: 'Умножение — это повторное сложение. Когда вы умножаете A на B, вы добавляете A к самому себе B раз. Представьте это как расположение объектов в строки и столбцы для формирования сетки.',
            div: 'Деление разбивает число на равные части. Когда вы делите A на B, вы находите, сколько групп B помещается в A. Думайте об этом как о равномерном распределении объектов.'
        }
    },
    ko: {
        langTitle: '언어 선택',
        menuTitle: '수학 기초',
        menuSubtitle: '3D로 기본 수학 연산을 배우세요',
        difficultyLabel: '난이도:',
        easy: '쉬움',
        medium: '보통',
        hard: '어려움',
        learn: '학습',
        learnDesc: '이론 및 3D 시각화',
        practice: '연습',
        practiceDesc: '대화형 연습',
        test: '테스트',
        testDesc: '점수가 있는 퀴즈',
        selectOperation: '연산 선택',
        addition: '덧셈',
        subtraction: '뺄셈',
        multiplication: '곱셈',
        division: '나눗셈',
        learning: '학습',
        practicing: '연습',
        testing: '테스트',
        theory: '이론',
        nextExample: '다음 예제',
        check: '확인',
        next: '다음',
        score: '점수:',
        time: '시간:',
        question: '문제:',
        testComplete: '테스트 완료!',
        correct: '정답',
        wrong: '오답',
        tryAgain: '다시 시도',
        menu: '메뉴',
        excellent: '훌륭합니다! 만점!',
        great: '잘했어요!',
        good: '좋은 노력!',
        keepPracticing: '계속 연습하세요!',
        theories: {
            add: '덧셈은 두 개 이상의 숫자를 하나의 합으로 결합합니다. 덧셈을 할 때 모든 양의 총합을 세는 것입니다. 이것을 객체 그룹을 하나의 더 큰 그룹으로 병합하는 것으로 시각화하세요.',
            sub: '뺄셈은 두 숫자의 차이를 찾습니다. 양을 빼면 얼마나 남는지 알려줍니다. 이것을 그룹에서 객체를 제거하는 것으로 생각하세요.',
            mul: '곱셈은 반복된 덧셈입니다. A를 B로 곱할 때 A를 B번 자신에게 더하는 것입니다. 이것을 객체를 행과 열로 배열하여 격자를 형성하는 것으로 시각화하세요.',
            div: '나눗셈은 숫자를 동등한 부분으로 나눕니다. A를 B로 나눌 때 A에 B 그룹이 몇 개 들어가는지 찾는 것입니다. 이것을 객체를 동등하게 분배하는 것으로 생각하세요.'
        }
    },
    ja: {
        langTitle: '言語を選択',
        menuTitle: '数学の基礎',
        menuSubtitle: '3Dで基本的な数学演算を学ぶ',
        difficultyLabel: '難易度:',
        easy: '簡単',
        medium: '普通',
        hard: '難しい',
        learn: '学習',
        learnDesc: '理論と3D視覚化',
        practice: '練習',
        practiceDesc: 'インタラクティブ演習',
        test: 'テスト',
        testDesc: 'スコア付きクイズ',
        selectOperation: '演算を選択',
        addition: '足し算',
        subtraction: '引き算',
        multiplication: '掛け算',
        division: '割り算',
        learning: '学習',
        practicing: '練習',
        testing: 'テスト',
        theory: '理論',
        nextExample: '次の例',
        check: '確認',
        next: '次へ',
        score: 'スコア:',
        time: '時間:',
        question: '問題:',
        testComplete: 'テスト完了!',
        correct: '正解',
        wrong: '不正解',
        tryAgain: '再挑戦',
        menu: 'メニュー',
        excellent: '素晴らしい!満点!',
        great: 'よくできました!',
        good: '良い努力!',
        keepPracticing: '練習を続けましょう!',
        theories: {
            add: '足し算は2つ以上の数を1つの合計に結合します。足し算をするとき、すべての量の合計を数えています。これをオブジェクトのグループを1つの大きなグループに統合することとして視覚化してください。',
            sub: '引き算は2つの数の差を求めます。量を取り除いたときに何が残るかを教えてくれます。これをグループからオブジェクトを取り除くことと考えてください。',
            mul: '掛け算は繰り返しの足し算です。AをBで掛けるとき、AをB回自身に加えています。これをオブジェクトを行と列に配置してグリッドを形成することとして視覚化してください。',
            div: '割り算は数を等しい部分に分けます。AをBで割るとき、Aの中にBのグループがいくつ入るかを見つけています。これをオブジェクトを均等に分配することと考えてください。'
        }
    },
    es: {
        langTitle: 'Seleccionar idioma',
        menuTitle: 'Matematicas basicas',
        menuSubtitle: 'Aprende operaciones matematicas fundamentales en 3D',
        difficultyLabel: 'Dificultad:',
        easy: 'Facil',
        medium: 'Medio',
        hard: 'Dificil',
        learn: 'Aprender',
        learnDesc: 'Teoria y visualizacion 3D',
        practice: 'Practicar',
        practiceDesc: 'Ejercicios interactivos',
        test: 'Prueba',
        testDesc: 'Examen con puntuacion',
        selectOperation: 'Seleccionar operacion',
        addition: 'Suma',
        subtraction: 'Resta',
        multiplication: 'Multiplicacion',
        division: 'Division',
        learning: 'Aprendiendo',
        practicing: 'Practicando',
        testing: 'Prueba',
        theory: 'Teoria',
        nextExample: 'Siguiente ejemplo',
        check: 'Verificar',
        next: 'Siguiente',
        score: 'Puntuacion:',
        time: 'Tiempo:',
        question: 'Pregunta:',
        testComplete: 'Prueba completada!',
        correct: 'Correcto',
        wrong: 'Incorrecto',
        tryAgain: 'Intentar de nuevo',
        menu: 'Menu',
        excellent: 'Excelente! Puntuacion perfecta!',
        great: 'Buen trabajo!',
        good: 'Buen esfuerzo!',
        keepPracticing: 'Sigue practicando!',
        theories: {
            add: 'La suma combina dos o mas numeros en una sola suma. Cuando sumas, estas contando el total de todas las cantidades juntas. Visualizalo como fusionar grupos de objetos en un grupo mas grande.',
            sub: 'La resta encuentra la diferencia entre dos numeros. Te dice cuanto queda cuando quitas una cantidad. Piensa en ello como quitar objetos de un grupo.',
            mul: 'La multiplicacion es suma repetida. Cuando multiplicas A por B, estas sumando A a si mismo B veces. Visualizalo como organizar objetos en filas y columnas para formar una cuadricula.',
            div: 'La division divide un numero en partes iguales. Cuando divides A por B, estas encontrando cuantos grupos de B caben en A. Piensa en ello como distribuir objetos equitativamente.'
        }
    }
};

// ============================================
// DITHERING SHADER
// ============================================
const DitherShader = {
    uniforms: {
        tDiffuse: { value: null },
        resolution: { value: new THREE.Vector2() },
        ditherStrength: { value: 0.15 }
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform vec2 resolution;
        uniform float ditherStrength;
        varying vec2 vUv;

        // Bayer 4x4 dithering matrix
        float bayer4x4(vec2 pos) {
            int x = int(mod(pos.x, 4.0));
            int y = int(mod(pos.y, 4.0));
            int index = x + y * 4;

            float matrix[16];
            matrix[0] = 0.0/16.0;   matrix[1] = 8.0/16.0;   matrix[2] = 2.0/16.0;   matrix[3] = 10.0/16.0;
            matrix[4] = 12.0/16.0;  matrix[5] = 4.0/16.0;   matrix[6] = 14.0/16.0;  matrix[7] = 6.0/16.0;
            matrix[8] = 3.0/16.0;   matrix[9] = 11.0/16.0;  matrix[10] = 1.0/16.0;  matrix[11] = 9.0/16.0;
            matrix[12] = 15.0/16.0; matrix[13] = 7.0/16.0;  matrix[14] = 13.0/16.0; matrix[15] = 5.0/16.0;

            for(int i = 0; i < 16; i++) {
                if(i == index) return matrix[i];
            }
            return 0.0;
        }

        void main() {
            vec4 color = texture2D(tDiffuse, vUv);

            // Convert to grayscale
            float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));

            // Apply dithering
            vec2 pixelPos = vUv * resolution;
            float dither = bayer4x4(pixelPos);

            // Threshold with dithering
            float threshold = gray + (dither - 0.5) * ditherStrength;
            float result = step(0.5, threshold);

            // Mix between original and dithered based on strength
            vec3 finalColor = mix(color.rgb, vec3(result), ditherStrength * 2.0);

            gl_FragColor = vec4(finalColor, color.a);
        }
    `
};

// ============================================
// APPLICATION STATE
// ============================================
const state = {
    language: 'en',
    difficulty: 'easy',
    mode: null,
    operation: null,
    currentQuestion: null,
    score: 0,
    totalQuestions: 10,
    currentQuestionNum: 0,
    timer: 0,
    timerInterval: null,
    ditherEnabled: true,
    scene: null,
    camera: null,
    renderer: null,
    composer: null,
    controls: null,
    objects: [],
    animationId: null
};

// ============================================
// UTILITY FUNCTIONS
// ============================================
function t(key) {
    const keys = key.split('.');
    let value = translations[state.language];
    for (const k of keys) {
        value = value?.[k];
    }
    return value || key;
}

function generateQuestion() {
    const ranges = {
        easy: { min: 1, max: 20 },
        medium: { min: -50, max: 100 },
        hard: { min: -100, max: 500 }
    };

    const range = ranges[state.difficulty];
    let a, b, answer, symbol;

    switch (state.operation) {
        case 'add':
            a = Math.floor(Math.random() * (range.max - range.min + 1)) + range.min;
            b = Math.floor(Math.random() * (range.max - range.min + 1)) + range.min;
            answer = a + b;
            symbol = '+';
            break;
        case 'sub':
            a = Math.floor(Math.random() * (range.max - range.min + 1)) + range.min;
            b = Math.floor(Math.random() * (range.max - range.min + 1)) + range.min;
            if (state.difficulty === 'easy') {
                if (b > a) [a, b] = [b, a];
            }
            answer = a - b;
            symbol = '-';
            break;
        case 'mul':
            const mulMax = state.difficulty === 'easy' ? 10 : state.difficulty === 'medium' ? 15 : 20;
            a = Math.floor(Math.random() * mulMax) + 1;
            b = Math.floor(Math.random() * mulMax) + 1;
            answer = a * b;
            symbol = 'x';
            break;
        case 'div':
            const divMax = state.difficulty === 'easy' ? 10 : state.difficulty === 'medium' ? 12 : 15;
            b = Math.floor(Math.random() * divMax) + 1;
            answer = Math.floor(Math.random() * divMax) + 1;
            a = b * answer;
            symbol = '/';
            break;
    }

    return { a, b, answer, symbol };
}

function generateOptions(correctAnswer) {
    const options = [correctAnswer];
    const variance = state.difficulty === 'easy' ? 5 : state.difficulty === 'medium' ? 10 : 20;

    while (options.length < 4) {
        const offset = Math.floor(Math.random() * variance * 2) - variance;
        const wrongAnswer = correctAnswer + offset;
        if (wrongAnswer !== correctAnswer && !options.includes(wrongAnswer)) {
            options.push(wrongAnswer);
        }
    }

    return options.sort(() => Math.random() - 0.5);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// ============================================
// THREE.JS SETUP
// ============================================
function initThreeJS(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Clear previous
    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
    }
    container.innerHTML = '';
    state.objects = [];

    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0xffffff);

    // Camera
    const aspect = container.clientWidth / container.clientHeight;
    state.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    state.camera.position.set(0, 5, 10);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(container.clientWidth, container.clientHeight);
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(state.renderer.domElement);

    // Post-processing with dithering
    state.composer = new EffectComposer(state.renderer);
    const renderPass = new RenderPass(state.scene, state.camera);
    state.composer.addPass(renderPass);

    const ditherPass = new ShaderPass(DitherShader);
    ditherPass.uniforms.resolution.value.set(container.clientWidth, container.clientHeight);
    ditherPass.enabled = state.ditherEnabled;
    state.composer.addPass(ditherPass);
    state.ditherPass = ditherPass;

    // Controls
    state.controls = new OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.minDistance = 5;
    state.controls.maxDistance = 30;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    state.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    state.scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(20, 20, 0x000000, 0xcccccc);
    state.scene.add(gridHelper);

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        state.camera.aspect = width / height;
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(width, height);
        state.composer.setSize(width, height);
        if (state.ditherPass) {
            state.ditherPass.uniforms.resolution.value.set(width, height);
        }
    });
    resizeObserver.observe(container);

    // Animation loop
    function animate() {
        state.animationId = requestAnimationFrame(animate);
        state.controls.update();

        // Rotate objects slightly
        state.objects.forEach(obj => {
            if (obj.userData.rotate) {
                obj.rotation.y += 0.005;
            }
        });

        if (state.ditherEnabled) {
            state.composer.render();
        } else {
            state.renderer.render(state.scene, state.camera);
        }
    }
    animate();
}

function clearScene() {
    state.objects.forEach(obj => {
        state.scene.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) {
                obj.material.forEach(m => m.dispose());
            } else {
                obj.material.dispose();
            }
        }
    });
    state.objects = [];
}

// ============================================
// 3D VISUALIZATIONS FOR EACH OPERATION
// ============================================
function createCube(x, y, z, size = 1, wireframe = false) {
    const geometry = new THREE.BoxGeometry(size, size, size);
    let material;

    if (wireframe) {
        material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true
        });
    } else {
        material = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            flatShading: true
        });
    }

    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(x, y, z);

    if (!wireframe) {
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000 }));
        cube.add(line);
    }

    return cube;
}

function createSphere(x, y, z, radius = 0.5, wireframe = false) {
    const geometry = new THREE.SphereGeometry(radius, 16, 16);
    let material;

    if (wireframe) {
        material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true
        });
    } else {
        material = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            flatShading: true
        });
    }

    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(x, y, z);

    if (!wireframe) {
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000 }));
        sphere.add(line);
    }

    return sphere;
}

function visualizeAddition(a, b) {
    clearScene();

    // Group A - left side (solid cubes)
    const displayA = Math.min(a, 10);
    for (let i = 0; i < displayA; i++) {
        const cube = createCube(-4 + (i % 5) * 1.2, 0.5 + Math.floor(i / 5) * 1.2, 0, 1, false);
        cube.userData.rotate = false;
        state.scene.add(cube);
        state.objects.push(cube);
    }

    // Plus sign
    const plusGeo = new THREE.BoxGeometry(0.2, 1.5, 0.2);
    const plusMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const plusV = new THREE.Mesh(plusGeo, plusMat);
    plusV.position.set(0, 1, 0);
    state.scene.add(plusV);
    state.objects.push(plusV);

    const plusH = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.2, 0.2), plusMat);
    plusH.position.set(0, 1, 0);
    state.scene.add(plusH);
    state.objects.push(plusH);

    // Group B - right side (solid cubes)
    const displayB = Math.min(b, 10);
    for (let i = 0; i < displayB; i++) {
        const cube = createCube(4 - (i % 5) * 1.2, 0.5 + Math.floor(i / 5) * 1.2, 0, 1, false);
        cube.userData.rotate = false;
        state.scene.add(cube);
        state.objects.push(cube);
    }

    // Animate merge after delay
    setTimeout(() => {
        const total = Math.min(a + b, 20);
        clearScene();

        for (let i = 0; i < total; i++) {
            const cols = Math.ceil(Math.sqrt(total));
            const x = (i % cols) * 1.2 - (cols * 1.2) / 2 + 0.6;
            const z = Math.floor(i / cols) * 1.2 - 2;
            const cube = createCube(x, 0.5, z, 1, false);
            cube.userData.rotate = true;
            state.scene.add(cube);
            state.objects.push(cube);
        }
    }, 2000);
}

function visualizeSubtraction(a, b) {
    clearScene();

    const displayA = Math.min(Math.abs(a), 15);
    const cubes = [];

    // Create initial cubes (wireframe style for subtraction)
    for (let i = 0; i < displayA; i++) {
        const cols = Math.ceil(Math.sqrt(displayA));
        const x = (i % cols) * 1.2 - (cols * 1.2) / 2 + 0.6;
        const z = Math.floor(i / cols) * 1.2 - 2;
        const cube = createCube(x, 0.5, z, 1, true);
        cube.userData.rotate = false;
        state.scene.add(cube);
        state.objects.push(cube);
        cubes.push(cube);
    }

    // Animate removal
    const removeCount = Math.min(Math.abs(b), cubes.length);
    let removed = 0;

    const removeInterval = setInterval(() => {
        if (removed >= removeCount) {
            clearInterval(removeInterval);
            return;
        }

        const cubeToRemove = cubes[cubes.length - 1 - removed];
        if (cubeToRemove) {
            // Fade out animation
            const fadeOut = () => {
                if (cubeToRemove.scale.x > 0.1) {
                    cubeToRemove.scale.multiplyScalar(0.9);
                    requestAnimationFrame(fadeOut);
                } else {
                    state.scene.remove(cubeToRemove);
                    const idx = state.objects.indexOf(cubeToRemove);
                    if (idx > -1) state.objects.splice(idx, 1);
                }
            };
            fadeOut();
        }
        removed++;
    }, 300);
}

function visualizeMultiplication(a, b) {
    clearScene();

    const displayA = Math.min(a, 8);
    const displayB = Math.min(b, 8);

    // Create grid (solid cubes for multiplication)
    for (let i = 0; i < displayA; i++) {
        for (let j = 0; j < displayB; j++) {
            const cube = createCube(
                i * 1.2 - (displayA * 1.2) / 2 + 0.6,
                0.5,
                j * 1.2 - (displayB * 1.2) / 2 + 0.6,
                0.9,
                false
            );
            cube.userData.rotate = false;
            cube.scale.set(0, 0, 0);
            state.scene.add(cube);
            state.objects.push(cube);

            // Staggered appear animation
            setTimeout(() => {
                const grow = () => {
                    if (cube.scale.x < 1) {
                        cube.scale.addScalar(0.1);
                        requestAnimationFrame(grow);
                    } else {
                        cube.scale.set(1, 1, 1);
                    }
                };
                grow();
            }, (i * displayB + j) * 100);
        }
    }

    // Add row/column labels
    const labelGeo = new THREE.PlaneGeometry(1, 0.5);
    const createLabel = (text, x, z) => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, 64, 32);
        ctx.fillStyle = 'black';
        ctx.font = 'bold 24px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 32, 16);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
        const mesh = new THREE.Mesh(labelGeo, material);
        mesh.position.set(x, 0.1, z);
        mesh.rotation.x = -Math.PI / 2;
        return mesh;
    };
}

function visualizeDivision(a, b) {
    clearScene();

    // Large sphere that will split (solid to wireframe transition)
    const mainSphere = createSphere(0, 2, 0, 2, false);
    mainSphere.userData.rotate = true;
    state.scene.add(mainSphere);
    state.objects.push(mainSphere);

    // Animate split after delay
    setTimeout(() => {
        clearScene();

        const result = Math.floor(a / b);
        const displayResult = Math.min(result, 12);
        const angle = (2 * Math.PI) / displayResult;
        const radius = 3;

        for (let i = 0; i < displayResult; i++) {
            const x = Math.cos(angle * i) * radius;
            const z = Math.sin(angle * i) * radius;
            // Parts become wireframe
            const sphere = createSphere(x, 1, z, 0.8, true);
            sphere.userData.rotate = true;
            state.scene.add(sphere);
            state.objects.push(sphere);
        }

        // Show remainder if any
        const remainder = a % b;
        if (remainder > 0 && remainder < 10) {
            for (let i = 0; i < remainder; i++) {
                const smallSphere = createSphere(i * 0.6 - 1, 3, 0, 0.3, true);
                smallSphere.userData.rotate = false;
                state.scene.add(smallSphere);
                state.objects.push(smallSphere);
            }
        }
    }, 2000);
}

function visualizeOperation(question) {
    switch (state.operation) {
        case 'add':
            visualizeAddition(question.a, question.b);
            break;
        case 'sub':
            visualizeSubtraction(question.a, question.b);
            break;
        case 'mul':
            visualizeMultiplication(question.a, question.b);
            break;
        case 'div':
            visualizeDivision(question.a, question.b);
            break;
    }
}

// ============================================
// UI FUNCTIONS
// ============================================
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function updateLanguage() {
    // Update all text elements
    document.getElementById('lang-title').textContent = t('langTitle');
    document.getElementById('menu-title').textContent = t('menuTitle');
    document.getElementById('menu-subtitle').textContent = t('menuSubtitle');
    document.getElementById('difficulty-label').textContent = t('difficultyLabel');
    document.getElementById('diff-easy').textContent = t('easy');
    document.getElementById('diff-medium').textContent = t('medium');
    document.getElementById('diff-hard').textContent = t('hard');
    document.getElementById('mode-learn').textContent = t('learn');
    document.getElementById('mode-learn-desc').textContent = t('learnDesc');
    document.getElementById('mode-practice').textContent = t('practice');
    document.getElementById('mode-practice-desc').textContent = t('practiceDesc');
    document.getElementById('mode-test').textContent = t('test');
    document.getElementById('mode-test-desc').textContent = t('testDesc');
    document.getElementById('operation-title').textContent = t('selectOperation');
    document.getElementById('op-add').textContent = t('addition');
    document.getElementById('op-sub').textContent = t('subtraction');
    document.getElementById('op-mul').textContent = t('multiplication');
    document.getElementById('op-div').textContent = t('division');
    document.getElementById('learn-title').textContent = t('learning');
    document.getElementById('theory-title').textContent = t('theory');
    document.getElementById('next-example').textContent = t('nextExample');
    document.getElementById('practice-title').textContent = t('practicing');
    document.getElementById('check-answer').textContent = t('check');
    document.getElementById('next-practice').textContent = t('next');
    document.getElementById('test-title').textContent = t('testing');
    document.getElementById('score-label').textContent = t('score');
    document.getElementById('timer-label').textContent = t('time');
    document.getElementById('progress-label').textContent = t('question');
    document.getElementById('results-title').textContent = t('testComplete');
    document.getElementById('correct-label').textContent = t('correct');
    document.getElementById('wrong-label').textContent = t('wrong');
    document.getElementById('time-label').textContent = t('time');
    document.getElementById('retry-test').textContent = t('tryAgain');
    document.getElementById('back-to-menu-results').textContent = t('menu');
}

function startLearnMode() {
    state.mode = 'learn';
    showScreen('learn-screen');

    // Set theory text
    document.getElementById('theory-text').textContent = t(`theories.${state.operation}`);

    // Initialize 3D
    initThreeJS('canvas-container');

    // Generate first example
    nextLearnExample();
}

function nextLearnExample() {
    state.currentQuestion = generateQuestion();
    const { a, b, answer, symbol } = state.currentQuestion;
    document.getElementById('example-equation').textContent = `${a} ${symbol} ${b} = ${answer}`;
    visualizeOperation(state.currentQuestion);
}

function startPracticeMode() {
    state.mode = 'practice';
    showScreen('practice-screen');
    initThreeJS('practice-canvas-container');
    nextPracticeQuestion();
}

function nextPracticeQuestion() {
    state.currentQuestion = generateQuestion();
    const { a, b, symbol } = state.currentQuestion;
    document.getElementById('practice-equation').textContent = `${a} ${symbol} ${b} = ?`;
    document.getElementById('practice-input').value = '';
    document.getElementById('practice-feedback').textContent = '';
    document.getElementById('practice-feedback').className = '';
    document.getElementById('check-answer').style.display = 'block';
    document.getElementById('next-practice').style.display = 'none';
    visualizeOperation(state.currentQuestion);
}

function checkPracticeAnswer() {
    const input = parseInt(document.getElementById('practice-input').value);
    const feedback = document.getElementById('practice-feedback');

    if (isNaN(input)) {
        feedback.textContent = '?';
        return;
    }

    if (input === state.currentQuestion.answer) {
        feedback.textContent = `${t('correct')}! ${state.currentQuestion.a} ${state.currentQuestion.symbol} ${state.currentQuestion.b} = ${state.currentQuestion.answer}`;
        feedback.className = 'feedback-correct';
    } else {
        feedback.textContent = `${t('wrong')}! ${t('correct')}: ${state.currentQuestion.answer}`;
        feedback.className = 'feedback-wrong';
    }

    document.getElementById('check-answer').style.display = 'none';
    document.getElementById('next-practice').style.display = 'block';
}

function startTestMode() {
    state.mode = 'test';
    state.score = 0;
    state.currentQuestionNum = 0;
    state.timer = 0;

    showScreen('test-screen');
    initThreeJS('test-canvas-container');

    // Start timer
    if (state.timerInterval) clearInterval(state.timerInterval);
    state.timerInterval = setInterval(() => {
        state.timer++;
        document.getElementById('timer-value').textContent = formatTime(state.timer);
    }, 1000);

    nextTestQuestion();
}

function nextTestQuestion() {
    state.currentQuestionNum++;

    if (state.currentQuestionNum > state.totalQuestions) {
        endTest();
        return;
    }

    state.currentQuestion = generateQuestion();
    const { a, b, symbol } = state.currentQuestion;
    document.getElementById('test-equation').textContent = `${a} ${symbol} ${b} = ?`;
    document.getElementById('progress-value').textContent = `${state.currentQuestionNum}/${state.totalQuestions}`;
    document.getElementById('score-value').textContent = `${state.score}/${state.currentQuestionNum - 1}`;

    // Generate options
    const options = generateOptions(state.currentQuestion.answer);
    const optionsContainer = document.getElementById('test-options');
    optionsContainer.innerHTML = '';

    options.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = opt;
        btn.addEventListener('click', () => selectTestAnswer(opt, btn));
        optionsContainer.appendChild(btn);
    });

    visualizeOperation(state.currentQuestion);
}

function selectTestAnswer(selected, btn) {
    const buttons = document.querySelectorAll('.option-btn');
    buttons.forEach(b => b.disabled = true);

    if (selected === state.currentQuestion.answer) {
        btn.classList.add('correct');
        state.score++;
    } else {
        btn.classList.add('wrong');
        buttons.forEach(b => {
            if (parseInt(b.textContent) === state.currentQuestion.answer) {
                b.classList.add('correct');
            }
        });
    }

    setTimeout(nextTestQuestion, 1000);
}

function endTest() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }

    showScreen('results-screen');

    document.getElementById('final-score-value').textContent = state.score;
    document.getElementById('final-score-total').textContent = `/${state.totalQuestions}`;
    document.getElementById('correct-count').textContent = state.score;
    document.getElementById('wrong-count').textContent = state.totalQuestions - state.score;
    document.getElementById('time-taken').textContent = formatTime(state.timer);

    // Result message
    const percentage = (state.score / state.totalQuestions) * 100;
    let message;
    if (percentage === 100) message = t('excellent');
    else if (percentage >= 80) message = t('great');
    else if (percentage >= 60) message = t('good');
    else message = t('keepPracticing');

    document.getElementById('results-message').textContent = message;
}

function toggleDithering() {
    state.ditherEnabled = !state.ditherEnabled;
    if (state.ditherPass) {
        state.ditherPass.enabled = state.ditherEnabled;
    }
}

// ============================================
// EVENT LISTENERS
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Language selection
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            state.language = btn.dataset.lang;
            updateLanguage();
            showScreen('menu-screen');
        });
    });

    // Change language button
    document.getElementById('change-lang-btn').addEventListener('click', () => {
        showScreen('language-screen');
    });

    // Difficulty buttons
    document.querySelectorAll('.diff-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.difficulty = btn.dataset.diff;
        });
    });

    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            state.mode = btn.dataset.mode;
            if (state.mode === 'test') {
                state.operation = ['add', 'sub', 'mul', 'div'][Math.floor(Math.random() * 4)];
                startTestMode();
            } else {
                showScreen('operation-screen');
            }
        });
    });

    // Operation buttons
    document.querySelectorAll('.op-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            state.operation = btn.dataset.op;
            if (state.mode === 'learn') {
                startLearnMode();
            } else if (state.mode === 'practice') {
                startPracticeMode();
            }
        });
    });

    // Back buttons
    document.getElementById('back-to-menu').addEventListener('click', () => showScreen('menu-screen'));
    document.getElementById('back-from-learn').addEventListener('click', () => {
        if (state.animationId) cancelAnimationFrame(state.animationId);
        showScreen('operation-screen');
    });
    document.getElementById('back-from-practice').addEventListener('click', () => {
        if (state.animationId) cancelAnimationFrame(state.animationId);
        showScreen('operation-screen');
    });
    document.getElementById('back-from-test').addEventListener('click', () => {
        if (state.timerInterval) clearInterval(state.timerInterval);
        if (state.animationId) cancelAnimationFrame(state.animationId);
        showScreen('menu-screen');
    });

    // Learn controls
    document.getElementById('next-example').addEventListener('click', nextLearnExample);
    document.getElementById('toggle-dither').addEventListener('click', toggleDithering);

    // Practice controls
    document.getElementById('check-answer').addEventListener('click', checkPracticeAnswer);
    document.getElementById('next-practice').addEventListener('click', nextPracticeQuestion);
    document.getElementById('toggle-dither-practice').addEventListener('click', toggleDithering);
    document.getElementById('practice-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            if (document.getElementById('check-answer').style.display !== 'none') {
                checkPracticeAnswer();
            } else {
                nextPracticeQuestion();
            }
        }
    });

    // Results controls
    document.getElementById('retry-test').addEventListener('click', startTestMode);
    document.getElementById('back-to-menu-results').addEventListener('click', () => showScreen('menu-screen'));

    // Initialize with language selection
    updateLanguage();
});
