# str_opt

Current version in the dev branch!

Elnézést kérek a magyar nyelvű leírásért.

A dev branchben található a működő verzió.

A teszteket google colab-on futtattam, illetve saját PC-n Anaconda3 környezetben.

A letöltött könyvtárakban az str_opt/steering-optimizer/steering_optimizer/envs/ mappában indított jupyter notebook segítségével tesztek és a hálózat futtathatóak a környezeten.

Mellékeltem egy teszt notebookot, ebben a környezetet teszteltem. Itt található a klasszikus optimalizálóval megtalálható megoldás is. A notebook csak tesztelésre szolgált, nincs kommentelve. -> StrOptEnv_test_notebook

A környezet a str_opt/steering-optimizer/steering_optimizer/envs/ mappában található optimizer_env néven. Itt a gym környezetnek megfelelő függvényekben, főként a step() függvényben található a rendszert leíró rész, amelyet optimalizálni szeretnénk.

Készítettem egy próba ügynök - hálózatot is, ami szintén jupyter notebookból futtatható, neve: custom_network_and_test-Copy1
A hálózat megértését kommentek segítik.

Továbbá található egy colab notebook is amelyben automatikusan letölti a szükséges környezetet githubról és futtatja rajta a stable baselines DQN ügynök algoritmust majd közli az eredményt. Itt sajnos a környezet hibája miatt általában megszakad a tanítás, a hibákat a környezetben fel kell tárni.



