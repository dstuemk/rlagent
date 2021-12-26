BUILD=build
RUN=run
DATE=${shell date --iso-8601}
TIME=${shell date +%H%M%S}
RUN_PARENT_DIR=${RUN}/${USER}
RUN_DIR=${RUN_PARENT_DIR}/${DATE}/${TIME}

all:
	@echo "No target 'all' available..."


${BUILD}:
	mkdir ${BUILD}

compile: ${BUILD}
	cd ${BUILD} && cmake -DCMAKE_BUILD_TYPE=Release ..
	$(MAKE) -C ${BUILD} all

run: compile
	mkdir -p ${RUN_DIR}/data
	git ls-files | tar Tzcf - ${RUN_DIR}/code.tgz
	cp -r ${BUILD} ${RUN_DIR}/${BUILD}
	./${RUN_DIR}/${BUILD}/rlagent -exec learn -wdir ${RUN_DIR}/data

clean:
	rm -rf ${BUILD}
