var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "nsta_req": {
    "name": "nsta_req",
    "structName": "glue_msg_native_state_req",
    "className": "GlueMsgNativeStateReq",
    "fields": [
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "nsta_res": {
    "name": "nsta_res",
    "structName": "glue_msg_native_state_res",
    "className": "GlueMsgNativeStateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "meta_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12088-0bee379";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`, `nativeStateStore` means `putNativeState` / `getNativeState` are available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true,
      nativeStateStore: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // the defaults are also the server's own (default_template_kwargs_keys/vals at load, as JSON texts), which is what
    // scoring renders with; the overlay keeps them on chat requests, whose own kwargs win
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        // the server's own defaults too, so every request it renders gets them (natlang's /v1/natlang/score items as
        // well as chat), as llama-server's --chat-template-kwargs does; each value travels as its JSON text, which is
        // how the server stores them
        default_template_kwargs_keys: params.default_template_kwargs ? Object.keys(params.default_template_kwargs) : void 0,
        default_template_kwargs_vals: params.default_template_kwargs ? Object.values(params.default_template_kwargs).map((value) => JSON.stringify(value)) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  /**
   * Store a natlang native state (natlang's store-file format, natlang.native-state/1) after the model is loaded, as the
   * server's PUT /v1/neuralese/native-states/:id does: the bytes are verified (layout, finite values, content ID), kept as
   * received and, for a state of this model's identity, registered so prompts consume it exactly.
   * @param state The state's bytes
   * @param id The ID the state must have (its content ID); omit to accept any
   * @returns The consumption mode and the state's metadata ({id, ...metadata})
   */
  putNativeState(state, id = "") {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        const result = yield this.proxy.wllamaAction(
          "native_state",
          { _name: "nsta_req", action: "put", id, filename }
        );
        if (!result.success) {
          throw new WllamaError("Failed to put native state", "inference_error");
        }
        return yield this.getSingleResult(result.req_id, "native state put");
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Read a stored native state (one this engine produced, or one put or loaded), as the server's
   * GET /v1/neuralese/native-states/:id does: its bytes exactly as stored, natlang's store-file format.
   * @param id The state's ID
   * @returns The state's bytes and its metadata ({id, ...metadata})
   */
  getNativeState(id) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const result = yield this.proxy.wllamaAction(
        "native_state",
        { _name: "nsta_req", action: "get", id, filename }
      );
      if (!result.success) {
        throw new WllamaError("Failed to get native state", "inference_error");
      }
      const state = yield this.proxy.slotFileRead(filename);
      return { state, meta: JSON.parse(result.meta_json) };
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
