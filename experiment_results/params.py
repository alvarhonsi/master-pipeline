
params = {
    "test" : {
        "dir" : "../tests/sanity_check/",
        "profiles" : ["tyxe-test"],
        "reruns" : 1,
        "models_title" : "Test",
    },
    "misspec-sin10" : {
        "dir" : "../experiments/misspec-sin10/",
        "profiles" : ["sin10-3x256-s03", "sin10-3x256-sl", "sin10-2x128-s03", "sin10-2x128-sl", "sin10-1x64-s03", "sin10-1x64-sl", "sin10-linear-s03", "sin10-linear-sl"],
        "reruns" : 10,
        "models_title" : "Sin10",
    },
    "misspec-msin10" : {
        "dir" : "../experiments/misspec-multisin10/",
        "profiles" : ["multisin10-3x64-s05", "multisin10-3x64-sl", "multisin10-2x32-s05", "multisin10-2x32-sl", "multisin10-1x16-s05", "multisin10-1x16-sl", "multisin10-linear-s05", "multisin10-linear-sl"],
        "reruns" : 10,
        "models_title" : "Multisin10",
    },
    "misspec-msin20" : {
        "dir" : "../experiments/misspec-multisin20/",
        "profiles" : ["multisin20-3x512-s05", "multisin20-3x512-sl", "multisin20-2x256-s05", "multisin20-2x256-sl", "multisin20-1x128-s05", "multisin20-1x128-sl", "multisin20-linear-s05", "multisin20-linear-sl"],
        "reruns" : 5,
        "models_title" : "Multisin20",
    },
    "sou-sin10" : {
        "dir" : "../experiments/sou-sin10/",
        "profiles" : ["sin10-3x256-s003", "sin10-3x256-s03", "sin10-3x256-s3", "sin10-3x256-sl"],
        "reruns" : 10,
        "models_title" : "Sin10",
    },
    "sou-msin10" : {
        "dir" : "../experiments/sou-multisin10/",
        "profiles" : ["multisin10-3x64-s005", "multisin10-3x64-s05", "multisin10-3x64-s5", "multisin10-3x64-sl"],
        "reruns" : 10,
        "models_title" : "Multisin10",
    },
    "sou-msin20" : {
        "dir" : "../experiments/sou-multisin20/",
        "profiles" : ["multisin20-3x512-s005", "multisin20-3x512-s05", "multisin20-3x512-s5", "multisin20-3x512-sl"],
        "reruns" : 5,
        "models_title" : "Multisin20",
    },
}